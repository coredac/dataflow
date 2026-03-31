//===- TaskDivisibilityAnalysisPass.cpp - Analyze task divisibility ----===//
//
// This pass analyzes each taskflow.task operation to determine whether its
// loop nest contains parallel loops that can be tiled for data-level
// parallelism (DLP).
//
// Task divisibility categories:
//   - divisible: Has at least one parallel loop (no loop-carried deps) with
//     trip_count > 1.  Can be tiled into sibling sub-tasks for runtime
//     configuration duplication.
//   - atomic: No exploitable parallel loops.  Must execute as a single
//     indivisible unit.
//
// The pass attaches an attribute to each taskflow.task:
//   divisibility_info = {
//     divisibility   : StringAttr       ("divisible" or "atomic")
//     parallel_dims  : DenseI32ArrayAttr (loop depth indices of parallel loops)
//     parallel_space : DenseI32ArrayAttr (trip counts of those parallel loops)
//   }
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Loop Nest Traversal Helpers
//===----------------------------------------------------------------------===//

// Collects the full loop nest starting from `outermost`, walking into
// perfectly and imperfectly nested loops (only follows the first nested
// affine.for at each level to form the "spine" of the nest).
static SmallVector<affine::AffineForOp>
collectLoopNest(affine::AffineForOp outermost) {
  SmallVector<affine::AffineForOp> nest;
  affine::AffineForOp current = outermost;

  while (current) {
    nest.push_back(current);

    // Look for a single nested affine.for in the body.
    affine::AffineForOp nested = nullptr;
    for (Operation &op : current.getBody()->getOperations()) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {
        if (nested) {
          // Multiple nested loops — stop descending (not a simple chain).
          nested = nullptr;
          break;
        }
        nested = for_op;
      }
    }
    current = nested;
  }

  return nest;
}

//===----------------------------------------------------------------------===//
// Per-Task Parallelism Analysis
//===----------------------------------------------------------------------===//

struct TaskParallelismInfo {
  StringRef divisibility;          // "divisible" or "atomic"
  SmallVector<int> parallel_dims;  // Loop depth indices of parallel loops.
  SmallVector<int> parallel_space; // Trip counts of parallel dims.
};

// Analyzes a single taskflow.task and determines its category.
static TaskParallelismInfo analyzeTask(TaskflowTaskOp task_op) {
  TaskParallelismInfo info;
  info.divisibility = "atomic"; // Default: no parallelism found.

  // Finds the outermost affine.for in the task body.
  affine::AffineForOp outermost_loop = nullptr;
  task_op.getBody().walk([&](affine::AffineForOp for_op) {
    // We want the outermost loop. Walk visits ops in pre-order,
    // so the first affine.for encountered at the top level is outermost.
    if (!outermost_loop) {
      // Check that this loop is at the top level of the task body
      // (its parent is the task's block, not another loop).
      if (for_op->getParentOp() == task_op.getOperation()) {
        outermost_loop = for_op;
      }
    }
  });

  if (!outermost_loop) {
    llvm::errs() << "[TaskDivisibilityAnalysis] Task " << task_op.getTaskName()
                 << ": no affine.for found, classified as atomic\n";
    return info;
  }

  // Collect the loop nest spine.
  SmallVector<affine::AffineForOp> loop_nest = collectLoopNest(outermost_loop);

  llvm::errs() << "[TaskDivisibilityAnalysis] Task " << task_op.getTaskName()
               << ": loop nest depth = " << loop_nest.size() << "\n";

  // Analyze each loop level for parallelism.
  for (size_t depth = 0; depth < loop_nest.size(); ++depth) {
    affine::AffineForOp loop = loop_nest[depth];

    // Check if the loop is parallel (including reduction-parallel).
    SmallVector<affine::LoopReduction> reductions;
    bool is_parallel = affine::isLoopParallel(loop, &reductions);

    // Get the trip count.
    std::optional<uint64_t> trip_count = affine::getConstantTripCount(loop);
    int64_t tc =
        trip_count.has_value() ? static_cast<int64_t>(*trip_count) : -1;

    llvm::errs() << "[TaskDivisibilityAnalysis]   depth " << depth
                 << ": parallel=" << is_parallel << ", trip_count=" << tc;
    if (!reductions.empty()) {
      llvm::errs() << " (with " << reductions.size() << " reductions)";
    }
    llvm::errs() << "\n";

    if (is_parallel && tc > 1) {
      info.parallel_dims.push_back(static_cast<int64_t>(depth));
      info.parallel_space.push_back(tc);
    }
  }

  // Classify based on whether any parallel dims were found.
  if (!info.parallel_dims.empty()) {
    info.divisibility = "divisible";
  }

  llvm::errs() << "[TaskDivisibilityAnalysis] Task " << task_op.getTaskName()
               << " -> " << info.divisibility;
  if (!info.parallel_dims.empty()) {
    llvm::errs() << ", parallel_dims=[";
    for (size_t i = 0; i < info.parallel_dims.size(); ++i) {
      if (i > 0)
        llvm::errs() << ",";
      llvm::errs() << info.parallel_dims[i];
    }
    llvm::errs() << "], parallel_space=[";
    for (size_t i = 0; i < info.parallel_space.size(); ++i) {
      if (i > 0)
        llvm::errs() << ",";
      llvm::errs() << info.parallel_space[i];
    }
    llvm::errs() << "]";
  }
  llvm::errs() << "\n";

  return info;
}

//===----------------------------------------------------------------------===//
// Task Divisibility Analysis Pass
//===----------------------------------------------------------------------===//

struct TaskDivisibilityAnalysisPass
    : public PassWrapper<TaskDivisibilityAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TaskDivisibilityAnalysisPass)

  StringRef getArgument() const final { return "task-divisibility-analysis"; }

  StringRef getDescription() const final {
    return "Analyzes task divisibility based on loop parallelism";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::errs() << "[TaskDivisibilityAnalysis] Running on function: "
                 << func.getName() << "\n";

    func.walk([&](TaskflowTaskOp task_op) {
      // Analyzes the task.
      TaskParallelismInfo info = analyzeTask(task_op);
      // Attaches the divisibility_info attribute to each task.
      OpBuilder builder(task_op);

      SmallVector<NamedAttribute, 3> div_attrs;
      task_op->setAttr("divisibility",
                       builder.getStringAttr(info.divisibility));
      task_op->setAttr("parallel_dims",
                       builder.getDenseI32ArrayAttr(info.parallel_dims));
      task_op->setAttr("parallel_space",
                       builder.getDenseI32ArrayAttr(info.parallel_space));
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::taskflow::createTaskDivisibilityAnalysisPass() {
  return std::make_unique<TaskDivisibilityAnalysisPass>();
}
