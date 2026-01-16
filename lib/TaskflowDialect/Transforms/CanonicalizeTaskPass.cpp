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
//===----------------------------------------------------------------------===//
// Memory Access Info
//===----------------------------------------------------------------------===//

struct MemoryAccessInfo {
  SetVector<Value> reads;
  SetVector<Value> writes;

  void analyze(TaskflowHyperblockOp hyperblock) {
    hyperblock.walk([&](Operation *op) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        reads.insert(load.getMemRef());
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        writes.insert(store.getMemRef());
      }
    });
  }

  SetVector<Value> getAllMemRefs() const {
    SetVector<Value> all;
    all.insert(reads.begin(), reads.end());
    all.insert(writes.begin(), writes.end());
    return all;
  }
};

//===----------------------------------------------------------------------===//
// Counter Collector
//===----------------------------------------------------------------------===//

class CounterCollector {
public:
  void collect(TaskflowHyperblockOp hyperblock) {
    for (Value idx : hyperblock.getIndices()) {
      collectRecursively(idx);
    }
  }

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

//===----------------------------------------------------------------------===//
// Block Argument Resolver
//===----------------------------------------------------------------------===//

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

  Value resolveToSource(Value val) const {
    auto it = blockArgToSource.find(val);
    return it != blockArgToSource.end() ? it->second : val;
  }

  Value getBlockArg(Value source) const {
    auto it = sourceToBlockArg.find(source);
    return it != sourceToBlockArg.end() ? it->second : Value();
  }

private:
  DenseMap<Value, Value> blockArgToSource;
  DenseMap<Value, Value> sourceToBlockArg;
};

//===----------------------------------------------------------------------===//
// Atomic Task Builder
//===----------------------------------------------------------------------===//

class AtomicTaskBuilder {
public:
  AtomicTaskBuilder(OpBuilder &builder, Location loc, unsigned global_task_idx,
                    DenseMap<Value, Value> &memref_to_latest_version)
      : builder(builder), loc(loc), global_task_idx(global_task_idx),
        memref_to_latest_version(memref_to_latest_version) {}

  TaskflowTaskOp build(TaskflowHyperblockOp hyperblock,
                       TaskflowTaskOp originalTask) {
    MemoryAccessInfo memInfo;
    memInfo.analyze(hyperblock);

    BlockArgResolver resolver(originalTask);

    // Determine task inputs
    SmallVector<Value> taskInputs;
    DenseMap<Value, unsigned> sourceToInputIdx;

    for (Value memref : memInfo.getAllMemRefs()) {
      Value source = resolver.resolveToSource(memref);
      Value inputVal = getLatestVersion(source);

      if (!sourceToInputIdx.count(source)) {
        sourceToInputIdx[source] = taskInputs.size();
        taskInputs.push_back(inputVal);
      }
    }

    // Determine task outputs
    SmallVector<Type> outputTypes;
    SmallVector<Value> writtenSources;

    for (Value memref : memInfo.writes) {
      Value source = resolver.resolveToSource(memref);
      outputTypes.push_back(source.getType());
      writtenSources.push_back(source);
    }

    // Create task
    std::string taskName = "Task_" + std::to_string(global_task_idx);
    auto newTask = builder.create<TaskflowTaskOp>(
        loc, outputTypes, TypeRange{}, taskInputs, ValueRange{},
        builder.getStringAttr(taskName));

    // Create task body
    Block *taskBody = new Block();
    newTask.getBody().push_back(taskBody);

    for (Value input : taskInputs) {
      taskBody->addArgument(input.getType(), loc);
    }

    // Build value mapping
    IRMapping mapping;

    for (auto [source, idx] : sourceToInputIdx) {
      BlockArgument newArg = taskBody->getArgument(idx);
      mapping.map(source, newArg);

      if (Value origArg = resolver.getBlockArg(source)) {
        mapping.map(origArg, newArg);
      }
    }

    // Clone counters and hyperblock
    OpBuilder taskBuilder(taskBody, taskBody->begin());
    cloneCounters(taskBuilder, hyperblock, mapping);
    cloneHyperblock(taskBuilder, hyperblock, mapping);

    // Create yield
    SmallVector<Value> yieldOperands;
    for (Value memref : memInfo.writes) {
      yieldOperands.push_back(mapping.lookupOrDefault(memref));
    }
    taskBuilder.setInsertionPointToEnd(taskBody);
    taskBuilder.create<TaskflowYieldOp>(loc, yieldOperands, ValueRange{});

    // Update latest versions
    auto outputs = newTask.getMemoryOutputs();
    for (auto [source, output] : llvm::zip(writtenSources, outputs)) {
      memref_to_latest_version[source] = output;
    }

    return newTask;
  }

private:
  Value getLatestVersion(Value source) {
    auto it = memref_to_latest_version.find(source);
    return it != memref_to_latest_version.end() ? it->second : source;
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
    SmallVector<Value> mappedIndices;
    for (Value idx : hyperblock.getIndices()) {
      mappedIndices.push_back(mapping.lookupOrDefault(idx));
    }

    SmallVector<Type> outputTypes(hyperblock.getOutputs().getTypes());
    auto newHB = taskBuilder.create<TaskflowHyperblockOp>(loc, outputTypes,
                                                          mappedIndices);

    Block *newBody = new Block();
    newHB.getBody().push_back(newBody);

    for (Value idx : mappedIndices) {
      newBody->addArgument(idx.getType(), loc);
    }

    Block *oldBody = &hyperblock.getBody().front();
    for (auto [oldArg, newArg] :
         llvm::zip(oldBody->getArguments(), newBody->getArguments())) {
      mapping.map(oldArg, newArg);
    }

    OpBuilder hbBuilder(newBody, newBody->begin());
    for (Operation &op : oldBody->without_terminator()) {
      hbBuilder.clone(op, mapping);
    }

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

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CanonicalizeTaskPass
    : public PassWrapper<CanonicalizeTaskPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeTaskPass)

  StringRef getArgument() const final { return "canonicalize-task"; }

  StringRef getDescription() const final {
    return "Canonicalizes tasks by splitting each hyperblock into a separate "
           "atomic task (one hyperblock per task)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<taskflow::TaskflowDialect, arith::ArithDialect,
                    memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();

    SmallVector<TaskflowTaskOp> tasks_to_process;
    func_op.walk(
        [&](TaskflowTaskOp task_op) { tasks_to_process.push_back(task_op); });

    unsigned global_task_idx = 0;

    for (TaskflowTaskOp original_task : tasks_to_process) {
      SmallVector<TaskflowHyperblockOp> hyperblocks;
      original_task.walk(
          [&](TaskflowHyperblockOp hb) { hyperblocks.push_back(hb); });

      assert(!hyperblocks.empty() &&
             "Expected at least one hyperblock in the task");

      if (hyperblocks.size() == 1) {
        continue;
      }

      //===----------------------------------------------------------------===//
      // Step 1: Build mapping from original task's memory outputs to their
      //         corresponding source memrefs (the original inputs).
      //===----------------------------------------------------------------===//

      // Get the yield operation to find which memrefs are yielded
      auto yield_op = cast<TaskflowYieldOp>(
          original_task.getBody().front().getTerminator());
      auto original_outputs = original_task.getMemoryOutputs();
      auto yielded_memrefs = yield_op.getMemoryResults();

      // Map: yielded block argument -> original task output
      DenseMap<Value, Value> yielded_to_output;
      for (auto [yielded, output] :
           llvm::zip(yielded_memrefs, original_outputs)) {
        yielded_to_output[yielded] = output;
      }

      // Map: original input memref -> original task output (if it's yielded)
      // This tells us which original outputs correspond to which input memrefs
      Block *orig_body = &original_task.getBody().front();
      auto orig_mem_inputs = original_task.getMemoryInputs();
      DenseMap<Value, Value> source_to_original_output;

      for (auto [input, arg] :
           llvm::zip(orig_mem_inputs, orig_body->getArguments())) {
        if (yielded_to_output.count(arg)) {
          source_to_original_output[input] = yielded_to_output[arg];
        }
      }

      //===----------------------------------------------------------------===//
      // Step 2: Create atomic tasks for each hyperblock.
      //===----------------------------------------------------------------===//

      DenseMap<Value, Value> memref_to_latest_version;
      OpBuilder builder(original_task);

      for (size_t i = 0; i < hyperblocks.size(); ++i) {
        AtomicTaskBuilder task_builder(builder, original_task.getLoc(),
                                       global_task_idx++,
                                       memref_to_latest_version);
        task_builder.build(hyperblocks[i], original_task);
      }

      //===----------------------------------------------------------------===//
      // Step 3: Replace uses of original task outputs with the latest versions.
      //===----------------------------------------------------------------===//

      for (auto [source, original_output] : source_to_original_output) {
        if (memref_to_latest_version.count(source)) {
          Value latest = memref_to_latest_version[source];
          original_output.replaceAllUsesWith(latest);
        }
      }

      //===----------------------------------------------------------------===//
      // Step 4: Erase the original task.
      //===----------------------------------------------------------------===//

      original_task.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createCanonicalizeTaskPass() {
  return std::make_unique<CanonicalizeTaskPass>();
}