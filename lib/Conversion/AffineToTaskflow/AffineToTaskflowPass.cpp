#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//------------------------------------------------------------------------------
// Helper Functions.
//------------------------------------------------------------------------------

// Collects memrefs that are loaded (read) within a given operation scope.
static void collectReadMemrefs(Operation *op, SetVector<Value> &read_memrefs) {
  op->walk([&](Operation *nested_op) {
    if (auto load_op = dyn_cast<affine::AffineLoadOp>(nested_op)) {
      read_memrefs.insert(load_op.getMemRef());
    } else if (auto load_op = dyn_cast<memref::LoadOp>(nested_op)) {
      read_memrefs.insert(load_op.getMemRef());
    }
  });
}

// Collects memrefs that are stored (written) within a given operation scope.
static void collectWrittenMemrefs(Operation *op,
                                  SetVector<Value> &written_memrefs) {
  op->walk([&](Operation *nested_op) {
    if (auto store_op = dyn_cast<affine::AffineStoreOp>(nested_op)) {
      written_memrefs.insert(store_op.getMemRef());
    } else if (auto store_op = dyn_cast<memref::StoreOp>(nested_op)) {
      written_memrefs.insert(store_op.getMemRef());
    }
  });
}

// Collects external values used within a given scope of operations.
static void collectExternalValues(Operation *root_op,
                                  const DenseSet<Operation *> &scope_ops,
                                  SetVector<Value> &external_values) {
  for (Value operand : root_op->getOperands()) {
    // Skips memref types (handled separately as memory dependencies).
    if (isa<MemRefType>(operand.getType())) {
      continue;
    }

    // Checks if it's a block argument.
    if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
      // Only adds if the block argument is not from within the scope.
      Operation *parent_op = block_arg.getOwner()->getParentOp();
      if (!scope_ops.contains(parent_op)) {
        external_values.insert(operand);
      }
      continue;
    }

    // Checks if the operand is defined outside the scope.
    Operation *def_op = operand.getDefiningOp();
    if (def_op && !scope_ops.contains(def_op)) {
      external_values.insert(operand);
    }
  }

  // Recursively processes nested operations.
  for (Region &region : root_op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        collectExternalValues(&op, scope_ops, external_values);
      }
    }
  }
}

// Updates operands of an operation using the value mapping.
static void
updateOperationOperands(Operation *op,
                        const DenseMap<Value, Value> &value_mapping) {
  for (OpOperand &operand : op->getOpOperands()) {
    Value original_value = operand.get();
    auto it = value_mapping.find(original_value);
    if (it != value_mapping.end()) {
      operand.set(it->second);
    }
  }
}

//------------------------------------------------------------------------------
// Task Conversion
//------------------------------------------------------------------------------

// Converts a top-level affine.for to a taskflow.task operation.
static TaskflowTaskOp convertLoopToTask(OpBuilder &builder,
                                        affine::AffineForOp for_op,
                                        DenseMap<Value, Value> &value_mapping,
                                        int task_id) {
  Location loc = for_op.getLoc();
  std::string task_name = "Task_" + std::to_string(task_id);

  // Collects all operations in the loop scope.
  DenseSet<Operation *> scope_ops;
  scope_ops.insert(for_op.getOperation());
  for_op.walk([&](Operation *op) { scope_ops.insert(op); });

  //-------------------------------------------------------------------
  // Step 1: Collects read and written memrefs.
  //-------------------------------------------------------------------
  SetVector<Value> read_memrefs;
  SetVector<Value> written_memrefs;
  collectReadMemrefs(for_op.getOperation(), read_memrefs);
  collectWrittenMemrefs(for_op.getOperation(), written_memrefs);

  llvm::errs() << "Read memrefs for loop:\n" << for_op << "\n";
  for (Value memref : read_memrefs) {
    llvm::errs() << memref << "\n";
  }

  llvm::errs() << "Written memrefs for loop:\n" << for_op << "\n";
  for (Value memref : written_memrefs) {
    llvm::errs() << memref << "\n";
  }

  //-------------------------------------------------------------------
  // Step 2: Determines memory inputs and outputs.
  //-------------------------------------------------------------------
  // Memory inputs: ALL memrefs that are accessed (read OR written).
  // This ensures WAR and WAW dependencies are respected.
  SetVector<Value> accessed_memrefs;
  accessed_memrefs.insert(read_memrefs.begin(), read_memrefs.end());
  accessed_memrefs.insert(written_memrefs.begin(), written_memrefs.end());

  // Memory outputs: ONLY memrefs that are written.
  // This ensures RAW and WAW dependencies are respected.
  SetVector<Value> output_memrefs;
  output_memrefs.insert(written_memrefs.begin(), written_memrefs.end());

  //-------------------------------------------------------------------
  // Step 3: Collects external SSA values (non-memref).
  //-------------------------------------------------------------------
  SetVector<Value> external_values;
  collectExternalValues(for_op.getOperation(), scope_ops, external_values);

  llvm::errs() << "External values for loop:\n" << for_op << "\n";
  for (Value val : external_values) {
    llvm::errs() << val << "\n";
  }

  //-------------------------------------------------------------------
  // Step 4: Resolves inputs through value mapping.
  //-------------------------------------------------------------------
  SmallVector<Value> memory_inputs;
  SmallVector<Value> value_inputs;
  IRMapping mapping;

  // Resolves memory inputs.
  for (Value memref : accessed_memrefs) {
    Value resolved_memref = value_mapping.lookup(memref);
    if (!resolved_memref) {
      resolved_memref = memref;
    }
    memory_inputs.push_back(resolved_memref);
    mapping.map(memref, resolved_memref);
  }

  // Resolves external SSA value inputs.
  for (Value external_val : external_values) {
    Value resolved_val = value_mapping.lookup(external_val);
    if (!resolved_val) {
      resolved_val = external_val;
    }
    value_inputs.push_back(resolved_val);
    mapping.map(external_val, resolved_val);
  }

  //-------------------------------------------------------------------
  // Step 5: Prepares output types.
  //-------------------------------------------------------------------
  SmallVector<Type> memory_output_types;
  for (Value memref : output_memrefs) {
    memory_output_types.push_back(memref.getType());
  }

  SmallVector<Type> value_output_types;
  for (Type result_type : for_op.getResultTypes()) {
    value_output_types.push_back(result_type);
  }

  //-------------------------------------------------------------------
  // Step 6: Creates the taskflow.task operation.
  //-------------------------------------------------------------------
  TaskflowTaskOp task_op = builder.create<TaskflowTaskOp>(
      loc,
      /*memory_outputs=*/memory_output_types,
      /*value_outputs=*/value_output_types,
      /*memory_inputs=*/memory_inputs,
      /*value_inputs=*/value_inputs,
      /*task_name=*/builder.getStringAttr(task_name));

  //-------------------------------------------------------------------
  // Step 7: Builds the task body.
  //-------------------------------------------------------------------
  Block *task_body = new Block();
  task_op.getBody().push_back(task_body);

  // Adds block arguments (memory inputs first, then value inputs).
  DenseMap<Value, BlockArgument> input_to_block_arg;
  // Memory input arguments.
  for (Value memref : accessed_memrefs) {
    BlockArgument arg = task_body->addArgument(memref.getType(), loc);
    mapping.map(memref, arg);
    input_to_block_arg[memref] = arg;
  }

  // Value input arguments.
  for (Value val : external_values) {
    BlockArgument arg = task_body->addArgument(val.getType(), loc);
    mapping.map(val, arg);
    input_to_block_arg[val] = arg;
  }

  // Clones loop into the task body.
  OpBuilder task_builder(task_body, task_body->begin());
  Operation *cloned_loop = task_builder.clone(*for_op.getOperation(), mapping);

  //---------------------------------------------------------------
  // Step 8: Creates the yield operation.
  //---------------------------------------------------------------
  task_builder.setInsertionPointToEnd(task_body);
  SmallVector<Value> memory_yield_operands;
  SmallVector<Value> value_yield_operands;

  // Memory yield outputs: yield the written memrefs.
  for (Value memref : output_memrefs) {
    if (input_to_block_arg.count(memref)) {
      memory_yield_operands.push_back(input_to_block_arg[memref]);
    } else {
      assert(false && "Written memref not in inputs!");
    }
  }

  // Value yield outputs: yield the loop results.
  for (Value result : cloned_loop->getResults()) {
    value_yield_operands.push_back(result);
  }
  task_builder.create<TaskflowYieldOp>(loc, memory_yield_operands,
                                       value_yield_operands);

  //-------------------------------------------------------------------
  // Step 9 : Updates value mapping with task outputs for subsequent tasks
  // conversion.
  //-------------------------------------------------------------------
  // Memory outputs.
  for (auto [memref, task_output] :
       llvm::zip(output_memrefs, task_op.getMemoryOutputs())) {
    value_mapping[memref] = task_output;
  }

  return task_op;
}

//------------------------------------------------------------------------------
// Main Conversion Process.
//------------------------------------------------------------------------------
// Converts a single function to TaskFlow operations.
static LogicalResult convertFuncToTaskflow(func::FuncOp func_op) {

  llvm::errs() << "\n===Converting function: " << func_op.getName() << "===\n";

  OpBuilder builder(func_op.getContext());
  SmallVector<affine::AffineForOp> loops_to_erase;
  DenseMap<Value, Value> value_mapping;
  int task_id_counter = 0;

  // Processes each block in the function.
  for (Block &block : func_op.getBlocks()) {
    // Collects operations to process (to avoid iterator invalidation).
    SmallVector<Operation *> ops_to_process;
    for (Operation &op : block) {
      ops_to_process.push_back(&op);
    }

    // Processes each operation in order (top to bottom).
    for (Operation *op : ops_to_process) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(op)) {
        // Converts affine.for to taskflow.task.
        OpBuilder builder(for_op);
        TaskflowTaskOp task_op = convertLoopToTask(
            builder, for_op, value_mapping, task_id_counter++);

        // Replaces uses of loop results with task value outputs.
        for (auto [loop_result, task_value_output] :
             llvm::zip(for_op.getResults(), task_op.getValueOutputs())) {
          loop_result.replaceAllUsesWith(task_value_output);
        }
        loops_to_erase.push_back(for_op);
      } else {
        // Updates operands of non-loop operations based on value_mapping.
        updateOperationOperands(op, value_mapping);
      }
    }
  }

  // Erases the original loops after conversion.
  for (affine::AffineForOp for_op : loops_to_erase) {
    for_op.erase();
  }

  return success();
}

class ConvertAffineToTaskflowPass
    : public PassWrapper<ConvertAffineToTaskflowPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAffineToTaskflowPass)

  StringRef getArgument() const final { return "convert-affine-to-taskflow"; }

  StringRef getDescription() const final {
    return "Convert Affine operations to Taskflow operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskflowDialect, affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
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

std::unique_ptr<Pass> mlir::createConvertAffineToTaskflowPass() {
  return std::make_unique<ConvertAffineToTaskflowPass>();
}