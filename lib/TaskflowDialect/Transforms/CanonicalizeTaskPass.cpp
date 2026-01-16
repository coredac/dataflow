#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Unit.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//---------------------------------------------------------------------------
// Memory Access Info: Information about memory accesses in a hyperblock.
//----------------------------------------------------------------------------
struct MemoryAccessInfo {
  SetVector<Value> reads;  // MemRefs that are read
  SetVector<Value> writes; // MemRefs that are written

  void analyze(TaskflowHyperblockOp hyperblock) {
    hyperblock.walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        reads.insert(load.getMemRef());
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        writes.insert(store.getMemRef());
      }
    });
  }

  // Get all memrefs (reads + writes, deduplicated)
  SetVector<Value> getAllMemRefs() const {
    SetVector<Value> all;
    all.insert(reads.begin(), reads.end());
    all.insert(writes.begin(), writes.end());
    return all;
  }
};

//---------------------------------------------------------------------------
// Counter Collector: Collects and sorts counter operations.
//----------------------------------------------------------------------------
class CounterCollector {
public:
  // Collect all counters needed by a hyperblock (including parents)
  void collect(TaskflowHyperblockOp hyperblock) {
    for (Value idx : hyperblock.getIndices()) {
      collectRecursively(idx);
    }
  }

  // Get counters sorted by depth (parents first)
  SmallVector<TaskflowCounterOp> getSortedCounters() const {
    SmallVector<TaskflowCounterOp> result(counters.begin(), counters.end());
    llvm::sort(result, [this](TaskflowCounterOp a, TaskflowCounterOp b) {
      return getDepth(a) < getDepth(b);
    });
    return result;
  }

private:
  void collectRecursively(Value idx) {
    auto counter = idx.getDefiningOp<TaskflowCounterOp>();
    if (!counter)
      return;

    counters.insert(counter);

    if (Value parent = counter.getParentIndex()) {
      collectRecursively(parent);
    }
  }

  size_t getDepth(TaskflowCounterOp counter) const {
    size_t depth = 0;
    Value parent = counter.getParentIndex();
    while (parent) {
      depth++;
      if (auto p = parent.getDefiningOp<TaskflowCounterOp>()) {
        parent = p.getParentIndex();
      } else {
        break;
      }
    }
    return depth;
  }

  SetVector<TaskflowCounterOp> counters;
};

//---------------------------------------------------------------------------
// Block Argument Resolver: Resolves block arguments to their source values.
//---------------------------------------------------------------------------
class BlockArgResolver {
public:
  explicit BlockArgResolver(TaskflowTaskOp task) {
    Block *body = &task.getBody().front();
    auto inputs = task.getMemoryInputs();
    auto args = body->getArguments();

    for (auto [input, arg] : llvm::zip(inputs, args)) {
      blockArgToSource[arg] = input;
      sourceToBlockArg[input] = arg;
    }
  }

  // Given a value (possibly a block arg), return the source memref
  Value resolveToSource(Value val) const {
    auto it = blockArgToSource.find(val);
    return it != blockArgToSource.end() ? it->second : val;
  }

  // Given a source memref, return the block argument
  Value getBlockArg(Value source) const {
    auto it = sourceToBlockArg.find(source);
    return it != sourceToBlockArg.end() ? it->second : Value();
  }

private:
  DenseMap<Value, Value> blockArgToSource;
  DenseMap<Value, Value> sourceToBlockArg;
};

//---------------------------------------------------------------------------
// Atomic Task Builder: Builds an atomic task from a single hyperblock.
//----------------------------------------------------------------------------
class AtomicTaskBuilder {
public:
  AtomicTaskBuilder(OpBuilder &builder, Location loc, unsigned global_task_idx,
                    DenseMap<Value, Value> &memref_to_latest_version)
      : builder(builder), loc(loc), global_task_idx(global_task_idx),
        memref_to_latest_version(memref_to_latest_version) {}
  TaskflowTaskOp build(TaskflowHyperblockOp hyperblock,
                       TaskflowTaskOp originalTask) {
    // Step 1: Analyze memory accesses
    MemoryAccessInfo memInfo;
    memInfo.analyze(hyperblock);

    // Step 2: Resolve block arguments to source memrefs
    BlockArgResolver resolver(originalTask);

    // Step 3: Determine task inputs (use latest versions)
    SmallVector<Value> taskInputs;
    DenseMap<Value, unsigned> sourceToInputIdx;

    for (Value memref : memInfo.getAllMemRefs()) {
      Value source = resolver.resolveToSource(memref);
      Value inputVal = getLatestVersion(source);

      // Avoid duplicates
      if (!sourceToInputIdx.count(source)) {
        sourceToInputIdx[source] = taskInputs.size();
        taskInputs.push_back(inputVal);
      }
    }

    // Step 4: Determine task outputs (written memrefs)
    SmallVector<Type> outputTypes;
    SmallVector<Value> writtenSources;

    for (Value memref : memInfo.writes) {
      Value source = resolver.resolveToSource(memref);
      outputTypes.push_back(source.getType());
      writtenSources.push_back(source);
    }

    // Step 5: Create the task operation
    std::string taskName = "Task_" + std::to_string(this->global_task_idx);
    auto newTask = builder.create<TaskflowTaskOp>(
        loc, outputTypes, TypeRange{}, taskInputs, ValueRange{},
        builder.getStringAttr(taskName));

    // Step 6: Create task body
    Block *taskBody = new Block();
    newTask.getBody().push_back(taskBody);

    for (Value input : taskInputs) {
      taskBody->addArgument(input.getType(), loc);
    }

    // Step 7: Build value mapping
    IRMapping mapping;

    // Map source memrefs -> new task's block arguments
    for (auto [source, idx] : sourceToInputIdx) {
      BlockArgument newArg = taskBody->getArgument(idx);
      mapping.map(source, newArg);

      // Also map original block arguments that refer to this source
      if (Value origArg = resolver.getBlockArg(source)) {
        mapping.map(origArg, newArg);
      }
    }

    // Step 8: Clone counters and hyperblock
    OpBuilder taskBuilder(taskBody, taskBody->begin());
    cloneCounters(taskBuilder, hyperblock, mapping);
    cloneHyperblock(taskBuilder, hyperblock, mapping);

    // Step 9: Create yield
    SmallVector<Value> yieldOperands;
    for (Value memref : memInfo.writes) {
      yieldOperands.push_back(mapping.lookupOrDefault(memref));
    }
    taskBuilder.setInsertionPointToEnd(taskBody);
    taskBuilder.create<TaskflowYieldOp>(loc, yieldOperands, ValueRange{});

    // Step 10: Update latest versions
    auto outputs = newTask.getMemoryOutputs();
    for (auto [source, output] : llvm::zip(writtenSources, outputs)) {
      this->memref_to_latest_version[source] = output;
    }

    return newTask;
  }

private:
  Value getLatestVersion(Value source) {
    auto it = this->memref_to_latest_version.find(source);
    return it != this->memref_to_latest_version.end() ? it->second : source;
  }

  void cloneCounters(OpBuilder &taskBuilder, TaskflowHyperblockOp hyperblock,
                     IRMapping &mapping) {
    CounterCollector collector;
    collector.collect(hyperblock);

    for (TaskflowCounterOp counter : collector.getSortedCounters()) {
      taskBuilder.clone(*counter.getOperation(), mapping);
    }
  }

  void cloneHyperblock(OpBuilder &taskBuilder, TaskflowHyperblockOp hyperblock,
                       IRMapping &mapping) {
    // Map indices
    SmallVector<Value> mappedIndices;
    for (Value idx : hyperblock.getIndices()) {
      mappedIndices.push_back(mapping.lookupOrDefault(idx));
    }

    // Create new hyperblock
    SmallVector<Type> outputTypes(hyperblock.getOutputs().getTypes());
    auto newHB = taskBuilder.create<TaskflowHyperblockOp>(loc, outputTypes,
                                                          mappedIndices);

    // Create body
    Block *newBody = new Block();
    newHB.getBody().push_back(newBody);

    for (Value idx : mappedIndices) {
      newBody->addArgument(idx.getType(), loc);
    }

    // Map old block args -> new block args
    Block *oldBody = &hyperblock.getBody().front();
    for (auto [oldArg, newArg] :
         llvm::zip(oldBody->getArguments(), newBody->getArguments())) {
      mapping.map(oldArg, newArg);
    }

    // Clone operations
    OpBuilder hbBuilder(newBody, newBody->begin());
    for (Operation &op : oldBody->without_terminator()) {
      hbBuilder.clone(op, mapping);
    }

    // Clone terminator
    if (auto yield =
            dyn_cast<TaskflowHyperblockYieldOp>(oldBody->getTerminator())) {
      SmallVector<Value> yieldOps;
      for (Value v : yield.getOutputs()) {
        yieldOps.push_back(mapping.lookupOrDefault(v));
      }
      hbBuilder.create<TaskflowHyperblockYieldOp>(loc, yieldOps);
    } else {
      hbBuilder.create<TaskflowHyperblockYieldOp>(loc, ValueRange{});
    }
  }
  OpBuilder &builder;
  Location loc;
  unsigned global_task_idx;
  DenseMap<Value, Value> &memref_to_latest_version;
};

//---------------------------------------------------------------------------
// Canonicalize Task Pass
//----------------------------------------------------------------------------
struct CanonicalizeTaskPass
    : public PassWrapper<CanonicalizeTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeTaskPass)

  StringRef getArgument() const final { return "canonicalize-task"; }

  StringRef getDescription() const final {
    return "Canonicalizes tasks by splitting each hyperblock into a separate "
           "atomic task (one hyperblock per task)";
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<taskflow::TaskflowDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();

    // Collects all tasks.
    SmallVector<TaskflowTaskOp> tasks_to_process;
    func_op.walk(
        [&](TaskflowTaskOp task_op) { tasks_to_process.push_back(task_op); });

    unsigned global_task_idx = 0;

    for (TaskflowTaskOp original_task : tasks_to_process) {
      // Collects hyperblocks.
      SmallVector<TaskflowHyperblockOp> hyperblocks;
      original_task.walk(
          [&](TaskflowHyperblockOp hb) { hyperblocks.push_back(hb); });
      assert(!hyperblocks.empty() &&
             "Expected at least one hyperblock in the task");
      if (hyperblocks.size() == 1) {
        // No need to canonicalize single-hyperblock tasks.
        continue;
      }

      // Tracks latest versions of memrefs for dependency chaining.
      DenseMap<Value, Value> memref_to_latest_version;

      // Creates atomic tasks for each hyperblock.
      OpBuilder builder(original_task);

      for (TaskflowHyperblockOp hb : hyperblocks) {
        AtomicTaskBuilder task_builder(builder, original_task.getLoc(),
                                       global_task_idx,
                                       memref_to_latest_version);
        task_builder.build(hb, original_task);
      }

      // Erases the original task.
      original_task.erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::taskflow::createCanonicalizeTaskPass() {
  return std::make_unique<CanonicalizeTaskPass>();
}