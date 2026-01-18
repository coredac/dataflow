// OptimizeTaskGraphPass.cpp - Optimizes Taskflow task graph.
//
// This pass performs the following optimizations on the Taskflow task graph:
// 1. Hyperblock Fusion: Merges hyperblocks with compatible counter structures.
// 2. Task Fusion: Merges producer-consumer tasks to reduce data transfer.
// 3. Dead Hyperblock Elimination: Removes unused hyperblocks.

#include "TaskflowDialect/Analysis/HyperblockDependencyAnalysis.h"
// #include "TaskflowDialect/Analysis/TaskDependencyAnalysis.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Resource Estimation (for future Architecture integration).
//===----------------------------------------------------------------------===//

/// Represents the estimated resource requirements for a hyperblock.
struct ResourceEstimate {
  int numOperations = 0;
  int numMemoryOps = 0;
  int numArithOps = 0;
};

/// Estimates the resource requirements for a hyperblock.
/// Used for resource constraint checking when Architecture is available.
[[maybe_unused]]
static ResourceEstimate estimateHyperblockResources(TaskflowHyperblockOp op) {
  ResourceEstimate estimate;
  op.getBody().walk([&](Operation *innerOp) {
    estimate.numOperations++;
    if (isa<memref::LoadOp, memref::StoreOp>(innerOp)) {
      estimate.numMemoryOps++;
    } else if (innerOp->getDialect()->getNamespace() == "arith") {
      estimate.numArithOps++;
    }
  });
  return estimate;
}

//===----------------------------------------------------------------------===//
// Hyperblock Fusion.
//===----------------------------------------------------------------------===//

/// Fuses two hyperblocks with identical counter structures.
/// The second hyperblock's operations are moved into the first hyperblock.
/// Handles SSA outputs by creating a new fused hyperblock.
static LogicalResult fuseHyperblocks(TaskflowHyperblockOp first,
                                      TaskflowHyperblockOp second,
                                      OpBuilder &builder) {
  // Verifies that the hyperblocks have the same indices.
  auto indicesFirst = first.getIndices();
  auto indicesSecond = second.getIndices();

  if (indicesFirst.size() != indicesSecond.size()) {
    return failure();
  }

  // Gets the blocks from both hyperblocks.
  Block &firstBlock = first.getBody().front();
  Block &secondBlock = second.getBody().front();

  // Finds the yield operations.
  auto firstYield = cast<TaskflowHyperblockYieldOp>(firstBlock.getTerminator());
  auto secondYield =
      cast<TaskflowHyperblockYieldOp>(secondBlock.getTerminator());

  // Creates a mapping from second's block arguments to first's block arguments.
  IRMapping mapping;
  for (size_t i = 0; i < indicesSecond.size(); ++i) {
    mapping.map(secondBlock.getArgument(i), firstBlock.getArgument(i));
  }

  // Sets insertion point before the first yield.
  builder.setInsertionPoint(firstYield);

  // Clones all operations from second (except the yield) into first.
  for (Operation &op : secondBlock.without_terminator()) {
    builder.clone(op, mapping);
  }

  // Merges outputs from both yields.
  SmallVector<Value> combinedOutputs;
  for (Value output : firstYield.getOutputs()) {
    combinedOutputs.push_back(output);
  }
  for (Value output : secondYield.getOutputs()) {
    // Maps the output through our mapping in case it references cloned values.
    Value mappedOutput = mapping.lookupOrDefault(output);
    combinedOutputs.push_back(mappedOutput);
  }

  // Replaces the first yield with a new one that has combined outputs.
  builder.setInsertionPoint(firstYield);
  builder.create<TaskflowHyperblockYieldOp>(firstYield.getLoc(),
                                             combinedOutputs);
  firstYield.erase();

  // Handles SSA outputs by creating a new hyperblock with combined
  // result types if either hyperblock has outputs.
  size_t firstOutputCount = first.getOutputs().size();
  size_t secondOutputCount = second.getOutputs().size();

  if (firstOutputCount > 0 || secondOutputCount > 0) {
    // Builds combined result types.
    SmallVector<Type> combinedResultTypes;
    for (Value res : first.getOutputs()) {
      combinedResultTypes.push_back(res.getType());
    }
    for (Value res : second.getOutputs()) {
      combinedResultTypes.push_back(res.getType());
    }

    // Creates a new hyperblock with the combined result types.
    builder.setInsertionPoint(first);
    auto newHyperblock = builder.create<TaskflowHyperblockOp>(
        first.getLoc(), combinedResultTypes, first.getIndices());

    // Moves the body from first to the new hyperblock.
    newHyperblock.getBody().takeBody(first.getBody());

    // Replaces uses of the original hyperblocks' results.
    for (size_t i = 0; i < firstOutputCount; ++i) {
      first.getOutputs()[i].replaceAllUsesWith(newHyperblock.getOutputs()[i]);
    }
    for (size_t i = 0; i < secondOutputCount; ++i) {
      second.getOutputs()[i].replaceAllUsesWith(
          newHyperblock.getOutputs()[firstOutputCount + i]);
    }

    // Erases both original hyperblocks.
    first.erase();
    second.erase();
  } else {
    // No outputs: simple case, just erase the second hyperblock.
    second.erase();
  }

  return success();
}

/// Attempts to fuse hyperblocks within a task.
/// Checks all pairs of hyperblocks and allows fusion
static void fuseHyperblocksInTask(TaskflowTaskOp taskOp,
                                   int maxBoundDiffForPeeling) {
  OpBuilder builder(taskOp.getContext());
  bool changed = true;

  // Iterates until no more fusions can be performed.
  while (changed) {
    changed = false;

    // Rebuilds the dependency graph after each fusion.
    HyperblockDependencyGraph depGraph;
    depGraph.buildFromTask(taskOp);

    const auto &hyperblocks = depGraph.getHyperblocks();
    if (hyperblocks.size() < 2) {
      return;
    }

    // Finds first fusable pair by checking all pairs (i, j) where i < j.
    bool foundPair = false;
    for (size_t i = 0; i < hyperblocks.size() && !foundPair; ++i) {
      for (size_t j = i + 1; j < hyperblocks.size() && !foundPair; ++j) {
        auto first = hyperblocks[i];
        auto second = hyperblocks[j];

        // Checks counter compatibility.
        if (!depGraph.areCountersCompatible(first, second,
                                            maxBoundDiffForPeeling)) {
          continue;
        }

        // Checks if fusion is safe (no circular dependencies would be created).
        // canFuse already checks for intermediate blocking dependencies.
        if (!depGraph.canFuse(first, second)) {
          continue;
        }

        // RAW dependency (first -> second) is safe to fuse because:
        // - We clone second's operations AFTER first's operations
        // - This preserves the original execution order
        // - Memory dependencies are satisfied
        //
        // Reverse dependency (second -> first) is NOT safe and is already
        // handled by canFuse() which checks program order.

        // Performs the fusion.
        llvm::errs() << "[OptimizeTaskGraph] Fusing hyperblocks at "
                     << first.getLoc() << " and " << second.getLoc() << "\n";

        if (succeeded(fuseHyperblocks(first, second, builder))) {
          changed = true;
          foundPair = true;
          // Restarts the loop with updated dependency graph.
        }
      }
    }
  }
}


//===----------------------------------------------------------------------===//
// Task Fusion (placeholder for future implementation).
//===----------------------------------------------------------------------===//

/// Fuses producer-consumer task pairs.
/// TODO: Implements actual task fusion logic.
[[maybe_unused]]
static void fuseProducerConsumerTasks(func::FuncOp funcOp) {
  // Task fusion is not yet implemented.
  // When enabled, this will:
  // 1. Build the task dependency graph.
  // 2. Find producer-consumer pairs.
  // 3. Check counter compatibility.
  // 4. Fuse compatible task pairs.
  (void)funcOp;
}

//===----------------------------------------------------------------------===//
// Dead Hyperblock Elimination.
//===----------------------------------------------------------------------===//

/// Checks if a hyperblock has no side effects that are used.
static bool isHyperblockDead(TaskflowHyperblockOp op) {
  // A hyperblock is considered dead if:
  // 1. It has no store operations, AND
  // 2. Its results (if any) are not used.

  bool hasStores = false;
  op.getBody().walk([&](memref::StoreOp storeOp) {
    hasStores = true;
  });

  if (hasStores) {
    return false;
  }

  // Checks if any results are used.
  for (Value result : op.getResults()) {
    if (!result.use_empty()) {
      return false;
    }
  }

  return true;
}

/// Eliminates dead hyperblocks from a function.
static void eliminateDeadHyperblocks(func::FuncOp funcOp) {
  SmallVector<TaskflowHyperblockOp> toErase;

  funcOp.walk([&](TaskflowHyperblockOp op) {
    if (isHyperblockDead(op)) {
      toErase.push_back(op);
    }
  });

  for (auto op : toErase) {
    op.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass Implementation.
//===----------------------------------------------------------------------===//

struct OptimizeTaskGraphPass
    : public PassWrapper<OptimizeTaskGraphPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeTaskGraphPass)

  OptimizeTaskGraphPass() = default;
  OptimizeTaskGraphPass(const OptimizeTaskGraphPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "optimize-task-graph"; }

  StringRef getDescription() const override {
    return "Optimizes Taskflow task graph by fusing hyperblocks and tasks.";
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Phase 1: Hyperblock Fusion.
    if (enableHyperblockFusion) {
      funcOp.walk([&](TaskflowTaskOp taskOp) {
        fuseHyperblocksInTask(taskOp, maxBoundDiffForPeeling);
      });
    }

    // Phase 2: Task Fusion.
    if (enableTaskFusion) {
      fuseProducerConsumerTasks(funcOp);
    }

    // Phase 3: Dead Hyperblock Elimination.
    eliminateDeadHyperblocks(funcOp);
  }

  Option<bool> enableHyperblockFusion{
      *this, "enable-hyperblock-fusion",
      llvm::cl::desc("Enables hyperblock fusion optimization."),
      llvm::cl::init(true)};

  Option<bool> enableTaskFusion{
      *this, "enable-task-fusion",
      llvm::cl::desc("Enables task fusion optimization (not yet implemented)."),
      llvm::cl::init(false)};

  Option<int> maxBoundDiffForPeeling{
      *this, "max-bound-diff",
      llvm::cl::desc("Specifies max loop bound difference for peeling."),
      llvm::cl::init(2)};
};

} // namespace

namespace mlir {
namespace taskflow {

/// Creates a pass that optimizes the task graph.
std::unique_ptr<Pass> createOptimizeTaskGraphPass() {
  return std::make_unique<OptimizeTaskGraphPass>();
}

} // namespace taskflow
} // namespace mlir
