#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//------------------------------------------------------------------------------
// Helper Functions.
//------------------------------------------------------------------------------
// Gets a descriptive task name based on the operation type.
static std::string generateTaskBaseName(Operation *op) {
  if (isa<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(op)) {
    return "conv2d";
  }
  if (isa<linalg::MatmulOp>(op)) {
    return "matmul";
  }
  if (isa<linalg::BatchMatmulOp>(op)) {
    return "batch_matmul";
  }
  if (isa<linalg::PoolingNchwMaxOp, linalg::PoolingNchwSumOp>(op)) {
    return "pooling";
  }
  if (auto generic_op = dyn_cast<linalg::GenericOp>(op)) {
    return "generic";
  }
  return "task";
}

// Maintains conversion context during the conversion process.
struct ConversionContext {
  // Maps original SSA values to task output values.
  DenseMap<Value, Value> value_mapping;

  // Maps graph input values to graph block arguments.
  DenseMap<Value, BlockArgument> graph_input_mapping;

  // Counter for generating unique task names.
  int task_counter = 0;

  // Generates a unique task name.
  std::string getTaskBaseName(StringRef base_name) {
    return (base_name + "_" + Twine(task_counter++)).str();
  }
};

// Operation classification.
static bool isComputeIntensiveOp(Operation *op) {
  // Returns true if the operation is one of the compute-intensive Linalg ops.
  return isa<linalg::Conv2DNchwFchwOp, linalg::MatmulOp, linalg::BatchMatmulOp,
             linalg::GenericOp, linalg::PoolingNchwMaxOp,
             linalg::PoolingNchwSumOp, tensor::PadOp>(op);
}

// Collects external values for a single operation.
static void collectExternalValuesForOp(
    Operation *op, const DenseSet<Operation *> &graph_op_set,
    func::FuncOp func_op, SetVector<Value> &external_values) {
  for (Value operand : op->getOperands()) {
    // Skips nested region block arguments.
    if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
      if (block_arg.getOwner()->getParentOp() != func_op.getOperation()) {
        continue;
      }
      external_values.insert(operand);
      continue;
    }

    // Skips values defined inside graph ops or nested regions.
    Operation *def_op = operand.getDefiningOp();
    if (def_op) {
      if (!graph_op_set.contains(def_op) &&
          def_op->getBlock()->getParentOp() == func_op.getOperation()) {
        external_values.insert(operand);
      }
    }
  }

  // Recurses into nested regions.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested_op : block) {
        collectExternalValuesForOp(&nested_op, graph_op_set, func_op,
                                   external_values);
      }
    }
  }
}

// Collects external values used by each graph operation.
static DenseMap<Operation *, SmallVector<Value>>
collectExternalValuesPerOp(ArrayRef<Operation *> graph_ops,
                           func::FuncOp func_op) {
  DenseSet<Operation *> graph_op_set(graph_ops.begin(), graph_ops.end());
  DenseMap<Operation *, SmallVector<Value>> op_external_values;

  for (Operation *op : graph_ops) {
    SetVector<Value> external_values;
    collectExternalValuesForOp(op, graph_op_set, func_op, external_values);
    op_external_values[op] =
        SmallVector<Value>(external_values.begin(), external_values.end());
  }

  return op_external_values;
}

//------------------------------------------------------------------------------
// Step 1: Scope Identification - Collects operations for the taskflow.graph
// op.
//------------------------------------------------------------------------------
// Collects all operations that should be included in the taskflow graph.
// Returns operations in topological order.
static SmallVector<Operation *> collectTaskflowGraphOps(func::FuncOp func_op) {
  SmallVector<Operation *> graph_ops;

  func_op.walk([&](Operation *op) {
    if (isComputeIntensiveOp(op)) {
      graph_ops.push_back(op);
    }
  });
  return graph_ops;
}

// Identifies external inputs to the taskflow graph (values defined outside the
// graph ops).
static SmallVector<Value> identifyGraphInputs(ArrayRef<Operation *> graph_ops,
                                              func::FuncOp func_op) {
  llvm::SetVector<Value> input_set;
  llvm::DenseSet<Operation *> graph_op_set(graph_ops.begin(), graph_ops.end());

  for (Operation *op : graph_ops) {
    collectExternalValuesForOp(op, graph_op_set, func_op, input_set);
  }

  return SmallVector<Value>(input_set.begin(), input_set.end());
}

// Identifies outputs from the graph (values used outside the graph ops).
static SmallVector<Value> identifyGraphOutputs(ArrayRef<Operation *> graph_ops,
                                               func::FuncOp func_op) {
  SmallVector<Value> outputs;
  DenseSet<Operation *> graph_op_set(graph_ops.begin(), graph_ops.end());

  for (Operation *op : graph_ops) {
    for (Value result : op->getResults()) {
      bool used_outside = false;
      for (Operation *user : result.getUsers()) {
        if (!graph_op_set.contains(user)) {
          used_outside = true;
          break;
        }
      }
      if (used_outside) {
        outputs.push_back(result);
      }
    }
  }
  return outputs;
}

//------------------------------------------------------------------------------
// Step 2: Task Contruction - Creates the taskflow.task ops.
//------------------------------------------------------------------------------
// Reolves the input value for a task operand.
// Returns the corresponding buffer value from the context, or wraps the
// original value.
static Value resolveTaskInput(OpBuilder &builder, Location loc,
                              Value original_value, ConversionContext &ctx) {
  // Checks if this value is produced by a task.
  if (ctx.value_mapping.count(original_value)) {
    return ctx.value_mapping[original_value];
  }

  // Checks if this value is a graph input.
  if (ctx.graph_input_mapping.count(original_value)) {
    return ctx.graph_input_mapping[original_value];
  }

  // Should not reach here for well-formed graphs.
  assert(false && "Unable to resolve task input value");
  return Value();
}

// Creates a taskflow.task op from a given operation.
// For pure data dependent workloads (e.g., AI workloads), taskes have:
//   - data_ins: input buffers
//   - data_outs: output buffers
//   - no control dependencies
static TaskflowTaskOp createTaskFromOp(OpBuilder &builder, Operation *op,
                                       ConversionContext &ctx,
                                       ArrayRef<Value> external_values) {
  Location loc = op->getLoc();
  std::string task_name = ctx.getTaskBaseName(generateTaskBaseName(op));

  // Resolves all external values to graph local values.
  SmallVector<Value> data_ins;
  IRMapping mapping;

  for (Value external_val : external_values) {
    Value resolved_input = resolveTaskInput(builder, loc, external_val, ctx);
    assert(resolved_input && "Failed to resolve task input");
    data_ins.push_back(resolved_input);
    mapping.map(external_val, resolved_input);
  }

  for (Value operand : op->getOperands()) {
    if (llvm::is_contained(external_values, operand)) {
      // Already mapped.
      continue;
    }
    Value resolved_input = resolveTaskInput(builder, loc, operand, ctx);
    assert(resolved_input && "Failed to resolve task input");
    data_ins.push_back(resolved_input);
    mapping.map(operand, resolved_input);
  }

  // Data outputs uses original result types.
  SmallVector<Type> data_out_types;
  for (Type result_type : op->getResultTypes()) {
    data_out_types.push_back(result_type);
  }

  // Creates the taskflow.task op.
  auto task_op = builder.create<TaskflowTaskOp>(
      loc,
      /*control_outs=*/TypeRange{},
      /*data_outs=*/data_out_types,
      /*control_ins=*/ValueRange{},
      /*data_ins=*/data_ins, builder.getStringAttr(task_name),
      /*indexing_maps=*/nullptr,
      /*iterator_types=*/nullptr);

  // Builds task body.
  Block *task_body = new Block();
  task_op.getBody().push_back(task_body);

  // Block arguments have same types as data_ins (original tensor types).
  for (Value input : data_ins) {
    task_body->addArgument(input.getType(), loc);
  }

  // Maps external values to task block arguments.
  for (size_t i = 0; i < external_values.size(); i++) {
    mapping.map(external_values[i], task_body->getArgument(i));
  }

  // Switches to the task body to clone the original operation.
  OpBuilder task_builder(task_body, task_body->begin());
  Operation *cloned_op = task_builder.clone(*op, mapping);
  // Yields the results.
  task_builder.create<TaskflowYieldOp>(loc, cloned_op->getResults());

  // Registers task outputs in context (same types as original results).
  for (auto [orig_result, task_output] :
       llvm::zip(op->getResults(), task_op.getDataOuts())) {
    ctx.value_mapping[orig_result] = task_output;
  }

  return task_op;
}

//------------------------------------------------------------------------------
// Step 3: Channel Insertion - Inserts taskflow.channel ops between tasks.
//------------------------------------------------------------------------------
static void insertChannels(OpBuilder &builder, ArrayRef<TaskflowTaskOp> tasks) {
  DenseSet<TaskflowTaskOp> task_set(tasks.begin(), tasks.end());

  for (TaskflowTaskOp producer_task : tasks) {
    Location loc = producer_task.getLoc();

    // For each data output of this producer task.
    for (Value data_out : producer_task.getDataOuts()) {
      // Collects all consumer tasks that use this output.
      SmallVector<std::pair<TaskflowTaskOp, OpOperand *>> consumer_tasks;

      for (OpOperand &use : data_out.getUses()) {
        Operation *user = use.getOwner();
        if (auto consumer_task = dyn_cast<TaskflowTaskOp>(user)) {
          if (task_set.contains(consumer_task)) {
            consumer_tasks.push_back({consumer_task, &use});
          }
        }
      }

      // Creates a dedicated channel for each consumer task.
      builder.setInsertionPointAfter(producer_task);

      for (auto [consumer_task, use] : consumer_tasks) {
        // Creates a new channel for this specific producer->consumer edge.
        auto channel_op = builder.create<TaskflowChannelOp>(
            loc, data_out.getType(), data_out);

        // Replaces only this specific use with the channel output.
        use->set(channel_op.getTarget());
      }
    }
  }
}

//------------------------------------------------------------------------------
// Step 4: Graph Construction - Creates the taskflow.graph op.
//------------------------------------------------------------------------------
static LogicalResult buildTaskflowGraph(
    OpBuilder &builder, func::FuncOp func_op, ArrayRef<Operation *> graph_ops,
    ArrayRef<Value> graph_inputs, MutableArrayRef<Value> graph_outputs,
    const DenseMap<Operation *, SmallVector<Value>> &op_external_values) {
  Location loc = func_op.getLoc();

  // Graph result types = original output types (no conversion).
  SmallVector<Type> result_types;
  for (Value output : graph_outputs) {
    result_types.push_back(output.getType());
  }

  // Creates graph op.
  auto graph_op =
      builder.create<TaskflowGraphOp>(loc, result_types, graph_inputs);

  // Builds graph body.
  Block *graph_body = new Block();
  graph_op.getBody().push_back(graph_body);

  // Block arguments have same types as graph inputs.
  ConversionContext ctx;
  for (Value input : graph_inputs) {
    BlockArgument arg = graph_body->addArgument(input.getType(), loc);
    ctx.graph_input_mapping[input] = arg;
  }

  // Converts each operation to a task.
  builder.setInsertionPointToStart(graph_body);
  SmallVector<TaskflowTaskOp> tasks;
  for (Operation *op : graph_ops) {
    const SmallVector<Value> &external_values = op_external_values.lookup(op);
    auto task_op = createTaskFromOp(builder, op, ctx, external_values);
    if (!task_op) {
      return failure();
    }
    tasks.push_back(task_op);
  }

  // Inserts channels between tasks.
  insertChannels(builder, tasks);

  // Creates graph return.
  SmallVector<Value> return_values;
  for (Value output : graph_outputs) {
    Value resolved = ctx.value_mapping[output];
    return_values.push_back(resolved);
  }
  builder.create<TaskflowReturnOp>(loc, return_values);

  // Replaces original outputs with graph results.
  for (auto [orig_output, graph_result] :
       llvm::zip(graph_outputs, graph_op.getResults())) {
    orig_output.replaceAllUsesExcept(graph_result, graph_op.getOperation());
  }

  // Erases original operations.
  for (Operation *op : llvm::reverse(graph_ops)) {
    op->erase();
  }

  return success();
}

//------------------------------------------------------------------------------
// Main Conversion Process.
//------------------------------------------------------------------------------
// Converts a single function to TaskFlow operations.
static LogicalResult convertFuncToTaskflow(func::FuncOp func_op) {
  // Step 1: Collects operations for the taskflow.graph op.
  SmallVector<Operation *> graph_ops = collectTaskflowGraphOps(func_op);
  if (graph_ops.empty()) {
    // No operations to convert.
    return success();
  }

  llvm::errs() << "Converting function: " << func_op.getName() << "\n";
  llvm::errs() << "Collected taskflow graph operations:\n";
  for (Operation *op : graph_ops) {
    llvm::errs() << "  " << *op << "\n";
  }

  SmallVector<Value> graph_inputs = identifyGraphInputs(graph_ops, func_op);
  SmallVector<Value> graph_outputs = identifyGraphOutputs(graph_ops, func_op);

  llvm::errs() << "Identified graph inputs:\n";
  for (Value input : graph_inputs) {
    llvm::errs() << "  " << input << "\n";
  }
  llvm::errs() << "Identified graph outputs:\n";
  for (Value output : graph_outputs) {
    llvm::errs() << "  " << output << "\n";
  }

  // Finds insertion point: after the last operation that defines a graph input.
  Operation *insertion_point = nullptr;
  for (Value input : graph_inputs) {
    if (auto *def_op = input.getDefiningOp()) {
      if (!insertion_point || insertion_point->isBeforeInBlock(def_op)) {
        insertion_point = def_op;
      }
    }
  }

  // Set the insertion point for the builder.
  OpBuilder builder(func_op.getContext());
  if (insertion_point) {
    builder.setInsertionPointAfter(insertion_point);
  } else {
    // If no inputs are defined by an operation (i.e., they are all function
    // arguments), insert the graph at the beginning of the function body.
    builder.setInsertionPointToStart(&func_op.front());
  }

  // Collects external values for each graph operation.
  DenseMap<Operation *, SmallVector<Value>> op_external_values =
      collectExternalValuesPerOp(graph_ops, func_op);

  // Step 2 & 3 & 4: Creates the taskflow.graph op.
  auto result = buildTaskflowGraph(builder, func_op, graph_ops, graph_inputs,
                                   graph_outputs, op_external_values);
  llvm::errs() << "Converted function to TaskFlow graph.\n";
  llvm::errs() << "Resulting function:\n";
  func_op.print(llvm::errs());
  llvm::errs() << "\n";

  return result;
}

class ConvertLinalgToTaskflowPass
    : public PassWrapper<ConvertLinalgToTaskflowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToTaskflowPass)

  StringRef getArgument() const final { return "convert-linalg-to-taskflow"; }

  StringRef getDescription() const final {
    return "Convert Linalg operations to Taskflow operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskflowDialect, linalg::LinalgDialect, func::FuncDialect,
                    arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    WalkResult result = module.walk([](func::FuncOp func_op) {
      if (failed(convertFuncToTaskflow(func_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertLinalgToTaskflowPass() {
  return std::make_unique<ConvertLinalgToTaskflowPass>();
}