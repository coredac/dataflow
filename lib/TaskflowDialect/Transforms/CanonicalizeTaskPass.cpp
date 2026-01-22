#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Unit.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//----------------------------------------------------------------------
// Memory and Value Access Info.
//----------------------------------------------------------------------
// This struct analyzes accesses information within a hyperblock.
struct AccessInfo {
  // Set of read memrefs.
  SetVector<Value> memref_reads;
  // Set of written memrefs.
  SetVector<Value> memref_writes;
  // Set of read values.
  SetVector<Value> value_reads;

  void analyze(TaskflowHyperblockOp hyperblock, Block *task_body) {
    DenseSet<Value> task_block_args;
    for (Value arg : task_body->getArguments()) {
      task_block_args.insert(arg);
    }

    hyperblock.walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        this->memref_reads.insert(load.getMemRef());
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        this->memref_writes.insert(store.getMemRef());
      }

      for (Value operand : op->getOperands()) {
        if (task_block_args.contains(operand)) {
          this->value_reads.insert(operand);
        }
      }
    });
  }

  SetVector<Value> getAllMemRefs() const {
    SetVector<Value> all;
    all.insert(this->memref_reads.begin(), this->memref_reads.end());
    all.insert(this->memref_writes.begin(), this->memref_writes.end());
    return all;
  }

  SetVector<Value> getAllValues() const { return this->value_reads; }
};

//----------------------------------------------------------------------
// Counter Collector.
//----------------------------------------------------------------------
// This class is used to collects all counters needed by a hyperblock.
class CounterCollector {
public:
  void collect(TaskflowHyperblockOp hyperblock) {
    for (Value idx : hyperblock.getIndices()) {
      collectRecursively(idx);
    }
  }

  // Gets the collected counters sorted by their depth.
  SmallVector<TaskflowCounterOp> getSortedCounters() const {
    SmallVector<TaskflowCounterOp> result(this->counters.begin(),
                                          this->counters.end());
    llvm::sort(result, [this](TaskflowCounterOp a, TaskflowCounterOp b) {
      return getDepth(a) < getDepth(b);
    });
    return result;
  }

private:
  // Collects counters recursively.
  void collectRecursively(Value idx) {
    TaskflowCounterOp counter = idx.getDefiningOp<TaskflowCounterOp>();
    if (!counter) {
      return;
    }
    this->counters.insert(counter);
    if (Value parent = counter.getParentIndex()) {
      collectRecursively(parent);
    }
  }

  // Gets the depth of a counter.
  size_t getDepth(TaskflowCounterOp counter) const {
    size_t depth = 0;
    Value parent = counter.getParentIndex();
    while (parent) {
      depth++;
      if (TaskflowCounterOp p = parent.getDefiningOp<TaskflowCounterOp>()) {
        parent = p.getParentIndex();
      } else {
        break;
      }
    }
    return depth;
  }

  SetVector<TaskflowCounterOp> counters;
};

//----------------------------------------------------------------------
// Block Argument Resolver.
//----------------------------------------------------------------------
// This class resolves the input arguments of a task block to their source
// values.
// For example:
// taskflow.task(%buf_input, %val_input) {
// ^bb0(%arg0: memref<?xi32>, %arg1: i32):   // ← block arguments
//   // %arg0 corresponds to %buf_input
//   // %arg1 corresponds to %val_input
// }
// resolveToSource(%arg0) -> %buf_input
class BlockArgResolver {
public:
  explicit BlockArgResolver(TaskflowTaskOp task) {
    Block *body = &task.getBody().front();

    // Resolves memory inputs.
    auto mem_inputs = task.getMemoryInputs();
    auto mem_args = body->getArguments().take_front(mem_inputs.size());
    for (auto [input, arg] : llvm::zip(mem_inputs, mem_args)) {
      this->block_arg_to_source[arg] = input;
      this->source_to_block_arg[input] = arg;
    }

    // Resolves value inputs.
    auto val_inputs = task.getValueInputs();
    auto val_args = body->getArguments().drop_front(mem_inputs.size());
    for (auto [input, arg] : llvm::zip(val_inputs, val_args)) {
      this->block_arg_to_source[arg] = input;
      this->source_to_block_arg[input] = arg;
    }
  }

  // Gets the source value for a given block argument.
  Value resolveToSource(Value val) const {
    auto it = this->block_arg_to_source.find(val);
    return it != this->block_arg_to_source.end() ? it->second : val;
  }

  // Gets the block argument for a given source value.
  Value getBlockArg(Value source) const {
    auto it = this->source_to_block_arg.find(source);
    return it != this->source_to_block_arg.end() ? it->second : Value();
  }

private:
  // Maps block argument to its source value.
  DenseMap<Value, Value> block_arg_to_source;
  // Maps source value to its block argument.
  DenseMap<Value, Value> source_to_block_arg;
};

//----------------------------------------------------------------------
// Atomic Task Builder.
//----------------------------------------------------------------------
// This class builds an atomic task from a hyperblock.
class AtomicTaskBuilder {
public:
  AtomicTaskBuilder(OpBuilder &builder, Location loc, unsigned global_task_idx,
                    DenseMap<Value, Value> &memref_to_latest_version,
                    DenseMap<Value, Value> &value_to_latest_version)
      : builder(builder), loc(loc), global_task_idx(global_task_idx),
        memref_to_latest_version(memref_to_latest_version),
        value_to_latest_version(value_to_latest_version) {}

  TaskflowTaskOp build(TaskflowHyperblockOp hyperblock,
                       TaskflowTaskOp original_task) {
    AccessInfo access_info;
    access_info.analyze(hyperblock, &original_task.getBody().front());

    BlockArgResolver resolver(original_task);

    // Determines memref inputs.
    SmallVector<Value> memref_inputs;
    DenseMap<Value, unsigned> source_to_memref_input_idx;

    for (Value memref : access_info.getAllMemRefs()) {
      Value source = resolver.resolveToSource(memref);
      Value input_memref = getLatestMemrefVersion(source);

      if (!source_to_memref_input_idx.count(source)) {
        source_to_memref_input_idx[source] = memref_inputs.size();
        memref_inputs.push_back(input_memref);
      }
    }

    // Determines value inputs.
    SmallVector<Value> value_inputs;
    DenseMap<Value, unsigned> source_to_value_input_idx;

    for (Value val : access_info.getAllValues()) {
      Value source = resolver.resolveToSource(val);
      Value input_val = getLatestValueVersion(source);

      if (!source_to_value_input_idx.count(source)) {
        source_to_value_input_idx[source] = value_inputs.size();
        value_inputs.push_back(input_val);
      }
    }

    // Determines memref outputs.
    SmallVector<Type> memref_output_types;
    // The source memrefs of the written memrefs.
    SmallVector<Value> written_memref_sources;

    for (Value memref : access_info.memref_writes) {
      Value source = resolver.resolveToSource(memref);
      memref_output_types.push_back(source.getType());
      written_memref_sources.push_back(source);
    }

    // Determines value outputs.
    SmallVector<Type> value_output_types;
    SmallVector<Value> yielded_value_sources;

    if (!hyperblock.getOutputs().empty()) {
      for (Value output : hyperblock.getOutputs()) {
        value_output_types.push_back(output.getType());
        // For value outputs, they are source themselves.
        yielded_value_sources.push_back(output);
      }
    }

    // Creates a new task.
    std::string task_name = "Task_" + std::to_string(this->global_task_idx);
    auto new_task = builder.create<TaskflowTaskOp>(
        this->loc, memref_output_types, value_output_types, memref_inputs,
        value_inputs, builder.getStringAttr(task_name));

    // Creates the task body.
    Block *task_body = new Block();
    new_task.getBody().push_back(task_body);

    // Adds memref input arguments.
    for (Value input : memref_inputs) {
      task_body->addArgument(input.getType(), this->loc);
    }
    // Adds value input arguments.
    for (Value input : value_inputs) {
      task_body->addArgument(input.getType(), this->loc);
    }

    // Builds value mapping.
    IRMapping mapping;

    // Maps memref inputs.
    for (auto [source, idx] : source_to_memref_input_idx) {
      BlockArgument new_arg = task_body->getArgument(idx);
      mapping.map(source, new_arg);

      if (Value orig_arg = resolver.getBlockArg(source)) {
        mapping.map(orig_arg, new_arg);
      }
    }

    // Maps value inputs.
    size_t value_arg_offset = memref_inputs.size();
    for (auto [source, idx] : source_to_value_input_idx) {
      BlockArgument new_arg = task_body->getArgument(value_arg_offset + idx);
      mapping.map(source, new_arg);

      if (Value orig_arg = resolver.getBlockArg(source)) {
        mapping.map(orig_arg, new_arg);
      }
    }

    // Clones counters and hyperblock.
    OpBuilder task_builder(task_body, task_body->begin());
    cloneCounters(task_builder, hyperblock, mapping);
    cloneHyperblock(task_builder, hyperblock, mapping);

    // Creates yield.
    SmallVector<Value> memref_yield_operands;
    for (Value memref : access_info.memref_writes) {
      memref_yield_operands.push_back(mapping.lookupOrDefault(memref));
    }

    SmallVector<Value> value_yield_operands;
    // If this hyperblock has value outputs, we need to yield them from the
    // mapped hyperblock.
    if (!hyperblock.getOutputs().empty()) {
      // Finds the cloned hyperblock op.
      TaskflowHyperblockOp cloned_hb = nullptr;
      for (Operation &op : task_body->getOperations()) {
        if (auto hb = dyn_cast<TaskflowHyperblockOp>(op)) {
          cloned_hb = hb;
          break;
        }
        if (cloned_hb) {
          for (Value output : cloned_hb.getOutputs()) {
            value_yield_operands.push_back(output);
          }
        }
      }
    }

    task_builder.setInsertionPointToEnd(task_body);
    task_builder.create<TaskflowYieldOp>(this->loc, memref_yield_operands,
                                         value_yield_operands);

    // Updates latest versions.
    auto memref_outputs = new_task.getMemoryOutputs();
    for (auto [source, output] :
         llvm::zip(written_memref_sources, memref_outputs)) {
      this->memref_to_latest_version[source] = output;
    }

    auto value_outputs = new_task.getValueOutputs();
    for (auto [source, output] :
         llvm::zip(yielded_value_sources, value_outputs)) {
      this->value_to_latest_version[source] = output;
    }

    return new_task;
  }

private:
  Value getLatestMemrefVersion(Value source) {
    auto it = this->memref_to_latest_version.find(source);
    return it != this->memref_to_latest_version.end() ? it->second : source;
  }

  Value getLatestValueVersion(Value source) {
    auto it = this->value_to_latest_version.find(source);
    return it != this->value_to_latest_version.end() ? it->second : source;
  }

  void cloneCounters(OpBuilder &task_builder, TaskflowHyperblockOp hyperblock,
                     IRMapping &mapping) {
    CounterCollector collector;
    collector.collect(hyperblock);

    for (TaskflowCounterOp counter : collector.getSortedCounters()) {
      task_builder.clone(*counter.getOperation(), mapping);
    }
  }

  void cloneHyperblock(OpBuilder &task_builder, TaskflowHyperblockOp hyperblock,
                       IRMapping &mapping) {
    SmallVector<Value> mapped_indices;
    for (Value idx : hyperblock.getIndices()) {
      mapped_indices.push_back(mapping.lookupOrDefault(idx));
    }

    SmallVector<Value> mapped_iter_args;
    for (Value arg : hyperblock.getIterArgs()) {
      mapped_iter_args.push_back(mapping.lookupOrDefault(arg));
    }

    SmallVector<Type> output_types(hyperblock.getOutputs().getTypes());
    auto newHB = task_builder.create<TaskflowHyperblockOp>(
        this->loc, output_types, mapped_indices, mapped_iter_args);

    Block *new_body = new Block();
    newHB.getBody().push_back(new_body);

    for (Value idx : mapped_indices) {
      new_body->addArgument(idx.getType(), this->loc);
    }

    for (Value arg : mapped_iter_args) {
      new_body->addArgument(arg.getType(), this->loc);
    }

    Block *old_body = &hyperblock.getBody().front();
    for (auto [old_arg, new_arg] :
         llvm::zip(old_body->getArguments(), new_body->getArguments())) {
      mapping.map(old_arg, new_arg);
    }

    OpBuilder hb_builder(new_body, new_body->begin());
    for (Operation &op : old_body->without_terminator()) {
      hb_builder.clone(op, mapping);
    }

    if (auto yield =
            dyn_cast<TaskflowHyperblockYieldOp>(old_body->getTerminator())) {
      SmallVector<Value> yield_results;
      SmallVector<Value> yield_iter_args_next;
      for (Value v : yield.getResults()) {
        yield_results.push_back(mapping.lookupOrDefault(v));
      }
      for (Value v : yield.getIterArgsNext()) {
        yield_iter_args_next.push_back(mapping.lookupOrDefault(v));
      }
      hb_builder.create<TaskflowHyperblockYieldOp>(this->loc, yield_results,
                                                   yield_iter_args_next);
    } else {
      hb_builder.create<TaskflowHyperblockYieldOp>(this->loc);
    }
  }

  OpBuilder &builder;
  Location loc;
  unsigned global_task_idx;
  DenseMap<Value, Value> &memref_to_latest_version;
  DenseMap<Value, Value> &value_to_latest_version;
};

//----------------------------------------------------------------------
// Pass Implementation.
//----------------------------------------------------------------------

struct CanonicalizeTaskPass
    : public PassWrapper<CanonicalizeTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeTaskPass)

  StringRef getArgument() const final { return "canonicalize-task"; }

  StringRef getDescription() const final {
    return "Canonicalizes tasks by splitting each hyperblock into a separate "
           "atomic task (one hyperblock per task)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<taskflow::TaskflowDialect, arith::ArithDialect,
                memref::MemRefDialect, func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();

    SmallVector<TaskflowTaskOp> tasks_to_process;
    func_op.walk(
        [&](TaskflowTaskOp task_op) { tasks_to_process.push_back(task_op); });

    unsigned global_task_idx = 0;

    for (TaskflowTaskOp original_task : tasks_to_process) {
      OpBuilder builder(original_task);
      // Collects hyperblocks within the original task.
      SmallVector<TaskflowHyperblockOp> hyperblocks;
      original_task.walk(
          [&](TaskflowHyperblockOp hb) { hyperblocks.push_back(hb); });

      assert(!hyperblocks.empty() &&
             "Expected at least one hyperblock in the task");

      // If there's only one hyperblock, it is already canonical.
      if (hyperblocks.size() == 1) {
        std::string task_name = "Task_" + std::to_string(global_task_idx++);
        original_task.setTaskNameAttr(builder.getStringAttr(task_name));
        continue;
      }

      //----------------------------------------------------------------
      // Step 1: Builds mapping from original task's memory outputs to their
      //         corresponding source memrefs (the original inputs).
      //----------------------------------------------------------------
      // Gets the yield operation to find which memrefs are yielded.
      auto yield_op = cast<TaskflowYieldOp>(
          original_task.getBody().front().getTerminator());
      auto original_mem_outputs = original_task.getMemoryOutputs();
      auto original_val_outputs = original_task.getValueOutputs();
      auto yielded_memrefs = yield_op.getMemoryResults();
      auto yielded_values = yield_op.getValueResults();

      // Map: yielded -> original task output.
      DenseMap<Value, Value> yielded_to_output;
      for (auto [yielded, output] :
           llvm::zip(yielded_memrefs, original_mem_outputs)) {
        yielded_to_output[yielded] = output;
      }
      for (auto [yielded, output] :
           llvm::zip(yielded_values, original_val_outputs)) {
        yielded_to_output[yielded] = output;
      }

      // Map: original input memref -> original task output (if it's yielded).
      // This tells us which original outputs correspond to which input memrefs.
      Block *orig_body = &original_task.getBody().front();
      auto orig_mem_inputs = original_task.getMemoryInputs();
      auto orig_val_inputs = original_task.getValueInputs();

      DenseMap<Value, Value> source_to_original_output;

      // Maps memref inputs.
      for (auto [input, arg] : llvm::zip(
               orig_mem_inputs,
               orig_body->getArguments().take_front(orig_mem_inputs.size()))) {
        if (yielded_to_output.count(arg)) {
          source_to_original_output[input] = yielded_to_output[arg];
        }
      }

      // Maps value inputs.
      for (auto [input, arg] : llvm::zip(
               orig_val_inputs,
               orig_body->getArguments().drop_front(orig_mem_inputs.size()))) {
        if (yielded_to_output.count(arg)) {
          source_to_original_output[input] = yielded_to_output[arg];
        }
      }

      //----------------------------------------------------------------
      // Step 2: Creates atomic tasks for each hyperblock.
      //----------------------------------------------------------------
      // Records the mapping from source memref to the latest version after
      // executing each atomic task.
      DenseMap<Value, Value> memref_to_latest_version;
      DenseMap<Value, Value> value_to_latest_version;

      for (size_t i = 0; i < hyperblocks.size(); ++i) {
        AtomicTaskBuilder task_builder(
            builder, original_task.getLoc(), global_task_idx++,
            memref_to_latest_version, value_to_latest_version);
        task_builder.build(hyperblocks[i], original_task);
      }

      //----------------------------------------------------------------
      // Step 3: Replaces uses of original task outputs with the latest
      // versions.
      //----------------------------------------------------------------
      for (auto [source, original_output] : source_to_original_output) {
        Value latest = nullptr;
        if (memref_to_latest_version.count(source)) {
          latest = memref_to_latest_version[source];
        } else if (value_to_latest_version.count(source)) {
          latest = value_to_latest_version[source];
        }

        if (latest) {
          original_output.replaceAllUsesWith(latest);
        }
      }

      //----------------------------------------------------------------
      // Step 4: Erase the original task.
      //----------------------------------------------------------------
      original_task.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createCanonicalizeTaskPass() {
  return std::make_unique<CanonicalizeTaskPass>();
}