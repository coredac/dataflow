#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//---------------------------------------------------------------------------
// Loop Info Structure
//----------------------------------------------------------------------------
struct LoopInfo {
  affine::AffineForOp for_op;
  int lower_bound;
  int upper_bound;
  int step;

  // For nested loops
  LoopInfo *parent_loop_info = nullptr;
  SmallVector<LoopInfo *> child_loops;

  // Generated counter index
  Value counter_index;
};

//----------------------------------------------------------------------------
// Helper Functions
//----------------------------------------------------------------------------
// Extracts loop parameters from affine.for operation.
static std::optional<LoopInfo> extractLoopBound(affine::AffineForOp for_op) {
  LoopInfo loop_info;
  loop_info.for_op = for_op;

  // Gets lower bound.
  if (for_op.hasConstantLowerBound()) {
    loop_info.lower_bound = for_op.getConstantLowerBound();
  } else {
    return std::nullopt;
  }

  // Gets upper bound.
  if (for_op.hasConstantUpperBound()) {
    loop_info.upper_bound = for_op.getConstantUpperBound();
  } else {
    return std::nullopt;
  }

  // Gets step.
  loop_info.step = for_op.getStepAsInt();

  return loop_info;
}

// Collects all affine.for loops and builds loop hierarchy.
static SmallVector<LoopInfo> collectLoopInfo(TaskflowTaskOp task_op) {
  SmallVector<LoopInfo> loops_info;
  DenseMap<Operation *, LoopInfo *> op_to_loopinfo;

  // Step 1: Collects all loops with its parameter.
  task_op.walk([&](affine::AffineForOp for_op) {
    auto info = extractLoopBound(for_op);
    if (!info) {
      assert(false && "Non-constant loop bounds are not supported.");
    }

    loops_info.push_back(*info);
    op_to_loopinfo[for_op.getOperation()] = &loops_info.back();
  });

  // Step 2: Builds parent-child relationships among loops.
  for (auto &loop_info : loops_info) {
    Operation *parent_op = loop_info.for_op->getParentOp();
    if (auto parent_for = dyn_cast<affine::AffineForOp>(parent_op)) {
      if (op_to_loopinfo.count(parent_for.getOperation())) {
        LoopInfo *parent_loop_info = op_to_loopinfo[parent_for.getOperation()];
        loop_info.parent_loop_info = parent_loop_info;
        parent_loop_info->child_loops.push_back(&loop_info);
      }
    }
  }

  return loops_info;
}

//----------------------------------------------------------------------------
// Counter Chain Creation
//----------------------------------------------------------------------------
// Recursively creates counter chain for each top-level loop.
static void createCounterChainRecursivly(OpBuilder &builder, Location loc,
                                         LoopInfo *loop_info,
                                         Value parent_counter) {}

// Creates counter chain for all top-level loops.
static void createCounterChain(OpBuilder &builder, Location loc,
                               SmallVector<LoopInfo *> &top_level_loops_info) {
  for (LoopInfo *loop_info : top_level_loops_info) {
    createCounterChainRecursivly(builder, loc, loop_info, nullptr);
  }
}

// Gets top-level loops' info (loops without parents).
static SmallVector<LoopInfo *>
getTopLevelLoopsInfo(SmallVector<LoopInfo> &loops_info) {
  SmallVector<LoopInfo *> top_level_loops_info;
  for (auto &loop_info : loops_info) {
    if (!loop_info.parent_loop_info) {
      top_level_loops_info.push_back(&loop_info);
    }
  }
  return top_level_loops_info;
}

//----------------------------------------------------------------------------
// Task Transformation
//----------------------------------------------------------------------------
// The main transformation function for TaskflowTaskOp.
static LogicalResult transformTask(TaskflowTaskOp task_op) {
  Location loc = task_op.getLoc();

  // Step 1: Collects loop information.
  SmallVector<LoopInfo> loops_info = collectLoopInfo(task_op);

  // Gets the body block of the task.
  Block *task_body = &task_op.getBody().front();

  // Finds the first loop in the task body.
  affine::AffineForOp first_loop_op = nullptr;
  for (Operation &op : task_body->getOperations()) {
    if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {
      first_loop_op = for_op;
      break;
    }
  }

  assert(first_loop_op && "No loops found in the task body.");

  // Step 2: Creates counter chain before the first loop.
  OpBuilder builder(first_loop_op);
  SmallVector<LoopInfo *> top_level_loops_info =
      getTopLevelLoopsInfo(loops_info);
  createCounterChain(builder, loc, top_level_loops_info);
  return success();
}

struct ConstructHyperblockFromTaskPass
    : public PassWrapper<ConstructHyperblockFromTaskPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstructHyperblockFromTaskPass)

  StringRef getArgument() const final {
    return "construct-hyperblock-from-task";
  }

  StringRef getDescription() const final {
    return "Constructs hyperblocks and counter chains from Taskflow tasks.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::taskflow::TaskflowDialect, affine::AffineDialect,
                    func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    // Collects all tasks.
    SmallVector<TaskflowTaskOp> tasks;
    func_op.walk([&](TaskflowTaskOp task_op) { tasks.push_back(task_op); });

    // Transforms each task.
    for (TaskflowTaskOp task_op : tasks) {
      llvm::errs() << "Number of tasks: " << tasks.size() << "\n";
      if (failed(transformTask(task_op))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::taskflow::createConstructHyperblockFromTaskPass() {
  return std::make_unique<ConstructHyperblockFromTaskPass>();
}