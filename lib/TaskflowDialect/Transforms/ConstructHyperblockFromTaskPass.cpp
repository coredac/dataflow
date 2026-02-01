#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//---------------------------------------------------------------------------
// Loop Info Structure.
//----------------------------------------------------------------------------
struct LoopInfo {
  affine::AffineForOp for_op;
  int lower_bound;
  int upper_bound;
  int step;

  // For nested loops.
  LoopInfo *parent_loop_info = nullptr;
  SmallVector<LoopInfo *> child_loops;

  // Generated counter index.
  Value counter_index;
};

//---------------------------------------------------------------------------
// Hyperblock Info Structure.
//----------------------------------------------------------------------------
// Represents a code block that should become a hyperblock.
struct HyperblockInfo {
  // The operations that belong to this hyperblock.
  SmallVector<Operation *> operations;

  // The counter indices that trigger this hyperblock (empty for top-level
  // operations before any loops).
  SmallVector<Value> trigger_indices;

  // Whether this hyperblock is nested within loops.
  bool is_loop_body = false;

  // The corresponding loop.
  affine::AffineForOp loop_op = nullptr;

  // Marks if this hyperblock follows the PLE pattern.
  bool is_ple_pattern = false;
};

//----------------------------------------------------------------------------
// Helper Functions.
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
// Counter Chain Creation.
//----------------------------------------------------------------------------
// Recursively creates counter chain for each top-level loop.
static void createCounterChainRecursivly(OpBuilder &builder, Location loc,
                                         LoopInfo *loop_info,
                                         Value parent_counter) {
  // Creates counter for this loop.
  Value counter_index;
  if (parent_counter) {
    // Nested counter.
    auto counter_op = builder.create<TaskflowCounterOp>(
        loc, builder.getIndexType(), parent_counter,
        builder.getIndexAttr(loop_info->lower_bound),
        builder.getIndexAttr(loop_info->upper_bound),
        builder.getIndexAttr(loop_info->step),
        /*Counter Type*/ nullptr, /*Counter ID*/ nullptr);
    counter_index = counter_op.getCounterIndex();
  } else {
    // Top-level counter.
    auto counter_op = builder.create<TaskflowCounterOp>(
        loc, builder.getIndexType(), /*parent_index=*/nullptr,
        builder.getIndexAttr(loop_info->lower_bound),
        builder.getIndexAttr(loop_info->upper_bound),
        builder.getIndexAttr(loop_info->step),
        /*Counter Type*/ nullptr, /*Counter ID*/ nullptr);
    counter_index = counter_op.getCounterIndex();
  }

  loop_info->counter_index = counter_index;

  // Recursively creates counters for child loops.
  for (LoopInfo *child : loop_info->child_loops) {
    createCounterChainRecursivly(builder, loc, child, counter_index);
  }
}

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
// Prologue-Loop-Epilogue Code (PLE) Pattern Detection
//----------------------------------------------------------------------------
// Prologue-Loop-Epilogue Code means code that appears before and after an inner
// loop. Example: for %i (outer loop) {
//   <prologue code>
//   for %j (nested loop) {
//     <loop body>
//   }
//   <epilogue code>  ← Loop-Epilogue Code
// }
// For this pattern, we need to wrap the inner loop and the prologue-epilogue
// code into a hyperblock. Only by doing this can we maintain the hyperblock as
// a pure data-driven code block.
struct PLEPattern {
  affine::AffineForOp outer_loop;
  affine::AffineForOp inner_loop;

  SmallVector<Operation *> prologue_code;
  SmallVector<Operation *> epilogue_code;

  bool has_ple_pattern = false;
};

// Detects Prologue-Loop-Epilogue Code pattern in the task.
static PLEPattern detectPLEPattern(affine::AffineForOp outer_loop) {
  PLEPattern pattern;
  pattern.has_ple_pattern = false;
  pattern.outer_loop = outer_loop;

  Block &body = outer_loop.getRegion().front();
  bool found_nested_loop = false;

  for (Operation &op : body.getOperations()) {
    if (auto nested_for = dyn_cast<affine::AffineForOp>(&op)) {
      found_nested_loop = true;
      if (!pattern.inner_loop) {
        pattern.inner_loop = nested_for;
      }
    } else if (!(isa<affine::AffineYieldOp>(&op) && op.getOperands().empty())) {
      if (!found_nested_loop) {
        pattern.prologue_code.push_back(&op);
      } else {
        pattern.epilogue_code.push_back(&op);
        pattern.has_ple_pattern = true;
      }
    }
  }

  if (found_nested_loop && (!pattern.prologue_code.empty())) {
    pattern.has_ple_pattern = true;
  }

  return pattern;
}

//----------------------------------------------------------------------------
// Hyperblock Creation
//----------------------------------------------------------------------------
// Recursively extracts hyperblocks from a region.
// Key insight: Operations in a loop body that are used by nested loops
// should be inlined into the nested loop's hyperblock.
static void extractHyperblocksInfoFromRegion(
    Region &region,
    const DenseMap<affine::AffineForOp, LoopInfo *> &loop_info_map,
    SmallVector<Value> parent_indices,
    SmallVector<HyperblockInfo> &hyperblocks_info,
    affine::AffineForOp enclosing_loop = nullptr,
    SmallVector<Operation *> inherited_ops = {}) {
  Block &block = region.front();
  SmallVector<Operation *> current_block_ops;

  current_block_ops.append(inherited_ops.begin(), inherited_ops.end());

  for (Operation &op : block.getOperations()) {
    if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {

      PLEPattern ple_pattern = detectPLEPattern(for_op);

      // Gets the loop info.
      LoopInfo *loop_info = loop_info_map.lookup(for_op);
      assert(loop_info && "Loop not found in loop_info_map");

      // Builds trigger indices for this loop (parent indices + this loop's
      // index).
      SmallVector<Value> loop_indices = parent_indices;
      loop_indices.push_back(loop_info->counter_index);

      // Handles the PLE pattern.
      if (ple_pattern.has_ple_pattern) {
        // 1. Emits any accumulated operations as a hyperblock.
        if (!current_block_ops.empty()) {
          HyperblockInfo info;
          info.operations = current_block_ops;
          info.trigger_indices = parent_indices;
          info.is_loop_body = !parent_indices.empty();
          info.loop_op = enclosing_loop;
          hyperblocks_info.push_back(info);
          current_block_ops.clear();
        }

        // 2. Creates a hyperblock for the prologue + inner loop + epilogue.
        HyperblockInfo info;
        if (!ple_pattern.prologue_code.empty()) {
          info.operations.append(ple_pattern.prologue_code.begin(),
                                 ple_pattern.prologue_code.end());
        }

        info.operations.push_back(ple_pattern.inner_loop);

        if (!ple_pattern.epilogue_code.empty()) {
          info.operations.append(ple_pattern.epilogue_code.begin(),
                                 ple_pattern.epilogue_code.end());
        }

        info.trigger_indices = loop_indices;
        info.is_loop_body = true;
        info.loop_op = for_op;
        info.is_ple_pattern = true;
        hyperblocks_info.push_back(info);

        // No need for further processing of this loop. Since we have already
        // handled the whole for_op.
        current_block_ops.clear();
        continue;
      }

      // Analyzes which of the current_ops are used by this loop.
      DenseSet<Value> values_used_in_loop;
      for_op.walk([&](Operation *nested_op) {
        for (Value operand : nested_op->getOperands()) {
          values_used_in_loop.insert(operand);
        }
      });

      SmallVector<Operation *> ops_for_nested_loop;
      SmallVector<Operation *> ops_not_used;
      bool used_by_loop = false;
      for (Operation *current_op : current_block_ops) {
        for (Value result : current_op->getResults()) {
          if (values_used_in_loop.contains(result)) {
            used_by_loop = true;
            break;
          }
        }
      }
      if (used_by_loop) {
        ops_for_nested_loop.append(current_block_ops.begin(),
                                   current_block_ops.end());
      } else {
        ops_not_used.append(current_block_ops.begin(), current_block_ops.end());
      }

      // Before processing the loop, emits any accumulated operations as a
      // hyperblock.
      if (!ops_not_used.empty()) {
        HyperblockInfo info;
        info.operations = ops_not_used;
        info.trigger_indices = parent_indices;
        info.is_loop_body = !parent_indices.empty();
        info.loop_op = enclosing_loop;
        hyperblocks_info.push_back(info);
      }

      // Recursively extracts hyperblocks from the loop body.
      extractHyperblocksInfoFromRegion(for_op.getRegion(), loop_info_map,
                                       loop_indices, hyperblocks_info, for_op,
                                       ops_for_nested_loop);
      current_block_ops.clear();
    } else if (isa<TaskflowYieldOp, TaskflowCounterOp>(&op) ||
               (isa<affine::AffineYieldOp>(&op) && op.getOperands().empty())) {
      // Skips TaskflowYieldOp, TaskflowCounterOp, and empty affine.yield.
      continue;
    } else {
      // Regular operation, accumulates it.
      current_block_ops.push_back(&op);
    }
  }

  // Emits any remaining operations as a hyperblock.
  if (!current_block_ops.empty()) {
    HyperblockInfo info;
    info.operations = current_block_ops;
    info.trigger_indices = parent_indices;
    info.is_loop_body = !parent_indices.empty();
    info.loop_op = enclosing_loop;
    hyperblocks_info.push_back(info);
    current_block_ops.clear();
  }
}

// Extracts all hyperblocks from a task.
static SmallVector<HyperblockInfo> extractHyperblocksInfo(
    TaskflowTaskOp task_op,
    const DenseMap<affine::AffineForOp, LoopInfo *> &loop_info_map) {
  SmallVector<HyperblockInfo> hyperblocks_info;
  // No parent indices for top-level hyperblocks (Not nested in a loop).
  SmallVector<Value> empty_indices;

  extractHyperblocksInfoFromRegion(task_op.getBody(), loop_info_map,
                                   empty_indices, hyperblocks_info);

  return hyperblocks_info;
}

// Collects all indices that are actually used by operations in the hyperblock.
static SmallVector<Value> collectUsedIndices(
    const SmallVector<Operation *> &operations,
    const SmallVector<Value> &candidate_indices,
    const DenseMap<affine::AffineForOp, LoopInfo *> &loop_info_map) {
  // Builds reverse mapping: counter -> induction variable.
  DenseMap<Value, Value> counter_to_indvar;
  for (auto [loop_op, loop_info] : loop_info_map) {
    counter_to_indvar[loop_info->counter_index] = loop_op.getInductionVar();
  }

  // Collects all values used by operations.
  SetVector<Value> used_indvars_set;
  for (Operation *op : operations) {
    for (Value operand : op->getOperands()) {
      used_indvars_set.insert(operand);
    }
  }

  // Returns in the same order as candidate_indices to maintain parent->child
  // order.
  SmallVector<Value> used_counters;
  for (Value counter : candidate_indices) {
    if (counter_to_indvar.count(counter)) {
      Value indvar = counter_to_indvar[counter];
      if (used_indvars_set.contains(indvar)) {
        used_counters.push_back(counter);
      }
    }
  }

  return used_counters;
}

// Determines output types for the hyperblock based on operations.
static SmallVector<Type>
determineHyperblockOutputTypes(const SmallVector<Operation *> &operations) {
  SmallVector<Type> output_types = {};

  // Checks if there's an affine.yield operation.
  for (Operation *op : operations) {
    if (auto affine_yield = dyn_cast<affine::AffineYieldOp>(op)) {
      // Uses the operand types of affine.yield as output types.
      for (Value operand : affine_yield.getOperands()) {
        output_types.push_back(operand.getType());
      }
      return output_types;
    }
  }

  // No affine.yield found, no output types needed.
  return output_types;
}

// Creates a taskflow.hyperblock operation from HyperblockInfo.
static TaskflowHyperblockOp createHyperblock(
    OpBuilder &builder, Location loc, HyperblockInfo &info, Block *task_body,
    const DenseMap<affine::AffineForOp, LoopInfo *> &loop_info_map) {
  // Collects only the indices that are actually used in the hyperblock.
  SmallVector<Value> used_indices =
      collectUsedIndices(info.operations, info.trigger_indices, loop_info_map);

  // Determines output types for the hyperblock based on operations.
  SmallVector<Type> output_types =
      determineHyperblockOutputTypes(info.operations);

  // Checks if there is a reduction in the hyperblock (with iter_args).
  SmallVector<Value> iter_args_init_values;
  bool is_reduction = false;
  if (info.loop_op && info.loop_op.getNumIterOperands() > 0) {
    is_reduction = true;
    for (Value init : info.loop_op.getInits()) {
      iter_args_init_values.push_back(init);
    }
  }
  // Creates the hyperblock operation.
  TaskflowHyperblockOp hyperblock_op;
  if (is_reduction) {
    hyperblock_op = builder.create<TaskflowHyperblockOp>(
        loc, output_types, used_indices, iter_args_init_values);
  } else {
    hyperblock_op = builder.create<TaskflowHyperblockOp>(
        loc, output_types, used_indices, /*iter_args=*/ValueRange{});
  }

  Block *hyperblock_body = new Block();
  hyperblock_op.getBody().push_back(hyperblock_body);

  // Adds block arguments for the used indices.
  for (Value idx : used_indices) {
    hyperblock_body->addArgument(idx.getType(), loc);
  }

  SmallVector<Value> iter_args_block_args;
  if (is_reduction) {
    for (Value init : iter_args_init_values) {
      BlockArgument arg = hyperblock_body->addArgument(init.getType(), loc);
      iter_args_block_args.push_back(arg);
    }
  }

  // Clone operations into the hyperblock body.
  OpBuilder hyperblock_builder(hyperblock_body, hyperblock_body->begin());
  IRMapping mapping;

  // Maps used indices to block arguments
  for (auto [idx, arg] :
       llvm::zip(used_indices, hyperblock_body->getArguments())) {
    mapping.map(idx, arg);
  }

  // Creates a mapping from loop counters to loop induction variables.
  DenseMap<Value, Value> counter_to_indvar;
  for (auto [loop_op, loop_info] : loop_info_map) {
    counter_to_indvar[loop_info->counter_index] = loop_op.getInductionVar();
  }

  // Maps loop induction variables to hyperblock block arguments.
  for (auto [idx, arg] :
       llvm::zip(used_indices, hyperblock_body->getArguments())) {
    if (counter_to_indvar.count(idx)) {
      Value indvar = counter_to_indvar[idx];
      mapping.map(indvar, arg);
    }
  }

  // If this hyperblock comes from a loop with iter_args, maps them.
  if (is_reduction) {
    Block &loop_body = info.loop_op.getRegion().front();
    auto loop_iter_args = loop_body.getArguments().drop_front(1);

    for (auto [loop_iter_arg, hb_iter_arg] :
         llvm::zip(loop_iter_args, iter_args_block_args)) {
      mapping.map(loop_iter_arg, hb_iter_arg);
    }
  }

  // Clones all operations and handle terminators.
  bool has_terminator = false;
  for (Operation *op : info.operations) {
    // Handles affine.yield specially - convert to hyperblock.yield.
    if (auto affine_yield = dyn_cast<affine::AffineYieldOp>(op)) {
      // Maps the yield operands through the IRMapping.
      SmallVector<Value> yield_operands;
      for (Value operand : affine_yield.getOperands()) {
        Value mapped_operand = mapping.lookupOrDefault(operand);
        yield_operands.push_back(mapped_operand);
      }

      // Creates hyperblock.yield with the mapped operands.
      hyperblock_builder.create<TaskflowHyperblockYieldOp>(loc, yield_operands,
                                                           yield_operands);
      has_terminator = true;
      continue;
    }

    // Clones regular operations.
    hyperblock_builder.clone(*op, mapping);
  }

  // Adds terminator if the last operation wasn't already a yield.
  if (!has_terminator) {
    hyperblock_builder.setInsertionPointToEnd(hyperblock_body);
    hyperblock_builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  MLIRContext *context = hyperblock_op.getContext();
  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);
  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         func::FuncDialect, taskflow::TaskflowDialect,
                         scf::SCFDialect>();
  target.addIllegalOp<affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineIfOp>();
  if (failed(
          applyPartialConversion(hyperblock_op, target, std::move(patterns)))) {
    assert(false && "Affine to Standard conversion failed.");
  }

  return hyperblock_op;
}

//----------------------------------------------------------------------------
// Task Transformation
//----------------------------------------------------------------------------
// The main transformation function for TaskflowTaskOp.
static LogicalResult transformTask(TaskflowTaskOp task_op) {
  Location loc = task_op.getLoc();

  // Step 1: Collects loop information.
  DenseMap<affine::AffineForOp, LoopInfo *> loop_info_map;
  SmallVector<LoopInfo> loops_info = collectLoopInfo(task_op);
  for (auto &loop_info : loops_info) {
    loop_info_map[loop_info.for_op] = &loop_info;
  }

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

  // Step 3: Extracts hyperblocks from task.
  SmallVector<HyperblockInfo> hyperblocks_info =
      extractHyperblocksInfo(task_op, loop_info_map);

  // Step 4: Creates taskflow.hyperblock operations for each hyperblock.
  builder.setInsertionPoint(first_loop_op);

  // Creates hyperblock ops.
  for (auto &info : hyperblocks_info) {
    TaskflowHyperblockOp hyperblock_op =
        createHyperblock(builder, loc, info, task_body, loop_info_map);

    // If this hyperblock has outputs and belongs to a loop with iter_args,
    // replace the loop results with the hyperblock outputs.
    if (info.loop_op && info.loop_op.getNumResults() > 0 &&
        (hyperblock_op.getNumResults() == info.loop_op.getNumResults())) {
      auto loop_results = info.loop_op.getResults();
      auto hyperblock_results = hyperblock_op.getOutputs();

      for (auto [loop_result, hb_result] :
           llvm::zip(loop_results, hyperblock_results)) {
        loop_result.replaceAllUsesWith(hb_result);
      }
    }
  }

  // Step 6: Collects and erases original loop operations.
  // Collects all operations to erase.
  SmallVector<Operation *> ops_to_erase;
  for (Operation &op : llvm::make_early_inc_range(task_body->getOperations())) {
    if (!isa<TaskflowYieldOp, TaskflowCounterOp, TaskflowHyperblockOp>(&op)) {
      ops_to_erase.push_back(&op);
    }
  }

  // Erases original operations.
  for (Operation *op : ops_to_erase) {
    op->erase();
  }

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
    registry
        .insert<mlir::taskflow::TaskflowDialect, affine::AffineDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    // Collects all tasks.
    SmallVector<TaskflowTaskOp> tasks;
    func_op.walk([&](TaskflowTaskOp task_op) { tasks.push_back(task_op); });

    // Transforms each task.
    for (TaskflowTaskOp task_op : tasks) {
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