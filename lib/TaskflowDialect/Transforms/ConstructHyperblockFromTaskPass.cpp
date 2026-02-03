#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
using namespace mlir;
using namespace mlir::taskflow;

namespace {

//==============================================================================
// Perfect Loop Band Detection.
//==============================================================================

// A perfect loop band is a sequence of perfectly nested loops where each loop
// (except the innermost) has exactly one child loop and no other operations
// (no prologue/epilogue).
struct PerfectLoopBand {
  // Outer to inner loop order.
  SmallVector<affine::AffineForOp> loops;

  bool isEmpty() const { return loops.empty(); }
  size_t getDepth() const { return loops.size(); }
};

// Detects the maximal perfect loop band starting from the given loop.
// Returns the sequence of perfectly nested loops.
static PerfectLoopBand detectPerfectLoopBand(affine::AffineForOp start_loop) {
  PerfectLoopBand band;
  affine::AffineForOp current_loop = start_loop;

  while (current_loop) {
    band.loops.push_back(current_loop);

    // Checks the body of current loop.
    Block &body = current_loop.getRegion().front();

    // Counts non-trivial operations (excluding yield).
    affine::AffineForOp nested_loop = nullptr;
    size_t num_loops = 0;
    size_t num_other_ops = 0;

    for (Operation &op : body) {
      if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {
        nested_loop = for_op;
        num_loops++;
      } else if (!(isa<affine::AffineYieldOp>(&op) &&
                   op.getNumOperands() == 0)) {
        num_other_ops++;
      }
    }

    // Perfect nesting condition: exactly 1 nested loop, no other operations.
    if (num_loops == 1 && num_other_ops == 0) {
      // Continues to next level.
      current_loop = nested_loop;
    } else {
      break; // Not perfect anymore.
    }
  }

  return band;
}

//==============================================================================
// Counter Creation.
//==============================================================================

struct CounterInfo {
  affine::AffineForOp loop;
  // The index value from taskflow.counter
  Value counter_index;
};

// Creates a chain of taskflow.counter operations for a perfect loop band.
// Returns counter info for each loop level.
static SmallVector<CounterInfo>
createCounterChain(OpBuilder &builder, Location loc,
                   const PerfectLoopBand &band) {
  SmallVector<CounterInfo> counters;
  Value parent_counter = nullptr;

  for (affine::AffineForOp loop : band.loops) {
    CounterInfo info;
    info.loop = loop;

    // Gets loop bounds.
    int32_t lb = 0, ub = 0, step = 0;
    if (loop.hasConstantLowerBound() && loop.hasConstantUpperBound()) {
      lb = loop.getConstantLowerBound();
      ub = loop.getConstantUpperBound();
      step = loop.getStepAsInt();
    } else {
      llvm::errs() << "Warning: Non-constant loop bounds not supported yet\n";
      continue;
    }

    // Creates counter.
    if (parent_counter) {
      // Creates nested counter with parent.
      TaskflowCounterOp counter_op = builder.create<TaskflowCounterOp>(
          loc,
          /*counter_index*/ builder.getIndexType(),
          /*parent_index*/ parent_counter,
          /*lower_bound*/ builder.getIndexAttr(lb),
          /*upper_bound*/ builder.getIndexAttr(ub),
          /*step*/ builder.getIndexAttr(step),
          /*counter_type*/ nullptr,
          /*counter_id*/ nullptr);
      info.counter_index = counter_op.getCounterIndex();
    } else {
      // Creates the top-level counter (no parent).
      TaskflowCounterOp counter_op = builder.create<TaskflowCounterOp>(
          loc,
          /*counter_index*/ builder.getIndexType(),
          /*parent_index*/ nullptr,
          /*lower_bound*/ builder.getIndexAttr(lb),
          /*upper_bound*/ builder.getIndexAttr(ub),
          /*step*/ builder.getIndexAttr(step),
          /*counter_type*/ nullptr,
          /*counter_id*/ nullptr);
      info.counter_index = counter_op.getCounterIndex();
    }

    parent_counter = info.counter_index;
    counters.push_back(info);
  }

  return counters;
}

//==============================================================================
// Hyperblock Creation.
//==============================================================================

// Analyzes which loop induction variables are actually used in the loop body.
// Returns indices of loops whose induction variables are used.
static SmallVector<size_t> analyzeUsedLoopIndices(const PerfectLoopBand &band) {
  SmallVector<size_t> used_indices;

  // Gets the deepest perfect loop's body.
  affine::AffineForOp deepest_loop = band.loops.back();
  Block &body = deepest_loop.getRegion().front();

  // Collects all values used in the body.
  DenseSet<Value> used_values;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      used_values.insert(operand);
    }
  });

  // Checks which loop induction variables are used.
  for (size_t i = 0; i < band.loops.size(); ++i) {
    affine::AffineForOp loop = band.loops[i];
    Value induction_var = loop.getInductionVar();
    if (used_values.contains(induction_var)) {
      used_indices.push_back(i);
    }
  }

  return used_indices;
}

// Clones the body of the deepest perfect loop in the perfect band into a
// hyperblock. Handles iter_args (reduction variables) by:
// 1. Adding iter_args initial values as hyperblock inputs
// 2. Mapping iter_args to hyperblock block arguments
// 3. Returning reduction results as hyperblock outputs
static TaskflowHyperblockOp
createHyperblockFromLoopBody(OpBuilder &builder, Location loc,
                             const PerfectLoopBand &band,
                             const SmallVector<CounterInfo> &counters) {
  // Gets the deepest perfect loop in the perfect nested band.
  affine::AffineForOp deepest_perfect_loop = band.loops.back();
  Block &loop_body = deepest_perfect_loop.getRegion().front();

  // Analyzes which loop indices are actually used.
  SmallVector<size_t> used_loop_indices = analyzeUsedLoopIndices(band);

  // Checks if the deepest loop has iter_args (reduction variables).
  bool has_iter_args = deepest_perfect_loop.getNumIterOperands() > 0;
  SmallVector<Value> iter_args_init_values = {};
  SmallVector<Type> iter_args_types = {};

  if (has_iter_args) {
    for (Value init_val : deepest_perfect_loop.getInits()) {
      iter_args_init_values.push_back(init_val);
      iter_args_types.push_back(init_val.getType());
    }
  }

  // Builds trigger values (only for USED counter indices)
  SmallVector<Value> trigger_values;
  for (size_t idx : used_loop_indices) {
    trigger_values.push_back(counters[idx].counter_index);
  }

  // Determines hyperblock result types (from iter_args if present).
  SmallVector<Type> result_types = {};
  if (has_iter_args) {
    result_types = iter_args_types;
  }

  // Creates hyperblock operation with iter_args as inputs.
  auto hyperblock_op = builder.create<TaskflowHyperblockOp>(
      loc, result_types, trigger_values, iter_args_init_values);

  // Builds block arguments:
  // 1. Counter indices (only for USED loop levels).
  // 2. Iter args values (passed through hyperblock invocation).
  SmallVector<Type> arg_types;
  SmallVector<Location> arg_locs;

  // Adds counter index arguments (only for used indices).
  for (size_t i = 0; i < used_loop_indices.size(); ++i) {
    arg_types.push_back(builder.getIndexType());
    arg_locs.push_back(loc);
  }

  // Adds iter_args as hyperblock block arguments.
  if (has_iter_args) {
    for (Type ty : iter_args_types) {
      arg_types.push_back(ty);
      arg_locs.push_back(loc);
    }
  }

  Block *hyperblock_body = &hyperblock_op.getBody().emplaceBlock();
  hyperblock_body->addArguments(arg_types, arg_locs);

  OpBuilder body_builder = OpBuilder::atBlockBegin(hyperblock_body);
  IRMapping mapper;

  // Maps USED loop induction variables to hyperblock arguments.
  for (size_t i = 0; i < used_loop_indices.size(); ++i) {
    size_t loop_idx = used_loop_indices[i];
    affine::AffineForOp loop = band.loops[loop_idx];
    mapper.map(loop.getInductionVar(), hyperblock_body->getArgument(i));
  }

  // Maps iter_args to hyperblock block arguments (after counter indices).
  if (has_iter_args) {
    for (size_t i = 0; i < iter_args_types.size(); ++i) {
      size_t arg_idx = used_loop_indices.size() + i;
      mapper.map(deepest_perfect_loop.getRegionIterArgs()[i],
                 hyperblock_body->getArgument(arg_idx));
    }
  }

  // Clones all operations from the deepest perfect loop's body.
  SmallVector<Value> yield_operands;

  for (Operation &op : loop_body) {
    // Handles affine.yield with operands (reduction results).
    if (auto yield_op = dyn_cast<affine::AffineYieldOp>(&op)) {
      if (yield_op.getNumOperands() > 0) {
        // Maps the yielded values for hyperblock's return.
        for (Value yielded : yield_op.getOperands()) {
          Value mapped = mapper.lookupOrDefault(yielded);
          yield_operands.push_back(mapped);
        }
      }
      continue; // Skips the yield itself.
    }

    // Clones operation (including nested affine.for with iter_args).
    Operation *cloned = body_builder.clone(op, mapper);

    // Updates mapper with cloned operation results.
    for (size_t i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), cloned->getResult(i));
    }
  }

  // Adds terminator with reduction results (if any).
  if (has_iter_args) {
    body_builder.create<TaskflowHyperblockYieldOp>(
        loc,
        /*iter_args_next=*/yield_operands, // No iter_args_next for final
                                           // iteration
        /*results=*/yield_operands);       // Reduction results
  } else {
    body_builder.create<TaskflowHyperblockYieldOp>(loc);
  }

  // Converts affine operations to standard/scf operations.
  MLIRContext *context = hyperblock_op.getContext();
  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);

  ConversionTarget target(*context);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         func::FuncDialect, TaskflowDialect, scf::SCFDialect>();
  target.addIllegalOp<affine::AffineLoadOp, affine::AffineStoreOp,
                      affine::AffineForOp, affine::AffineIfOp,
                      affine::AffineYieldOp>();

  if (failed(
          applyPartialConversion(hyperblock_op, target, std::move(patterns)))) {
    llvm::errs()
        << "Error: Failed to convert affine operations to standard/scf\n";
    return nullptr;
  }

  return hyperblock_op;
}

//============================================================================
// Task Transformation.
//===========================================================================
static LogicalResult transformTask(TaskflowTaskOp task_op) {
  Location loc = task_op.getLoc();
  Block &task_body = task_op.getBody().front();

  // Finds all top-level loops in the task.
  SmallVector<affine::AffineForOp> top_level_loops;
  for (Operation &op : task_body) {
    if (auto for_op = dyn_cast<affine::AffineForOp>(&op)) {
      top_level_loops.push_back(for_op);
    }
  }

  if (top_level_loops.empty()) {
    llvm::errs() << "No loops found in task " << task_op.getTaskName() << "\n";
    return success();
  }

  assert(top_level_loops.size() == 1 &&
         "Expected exactly one top-level loop in each task.");

  OpBuilder builder(&task_body, task_body.begin());

  // Stores mapping from loop results to hyperblock results.
  DenseMap<Value, Value> loop_result_to_hyperblock_result;

  // Processes each top-level loop.
  for (affine::AffineForOp top_loop : top_level_loops) {
    llvm::errs() << "\n[ConstructHyperblock] Processing top-level loop\n";

    // Step 1: Detects maximal perfect loop band.
    PerfectLoopBand band = detectPerfectLoopBand(top_loop);
    llvm::errs() << "  Detected perfect loop band of depth " << band.getDepth()
                 << "\n";

    // Step 2: Creates counter chain for the perfect band.
    builder.setInsertionPoint(top_loop);
    SmallVector<CounterInfo> counters = createCounterChain(builder, loc, band);
    llvm::errs() << "  Created " << counters.size() << " counters\n";

    // Step 3: Creates hyperblock from deepest loop's body.
    TaskflowHyperblockOp hyperblock_op =
        createHyperblockFromLoopBody(builder, loc, band, counters);
    llvm::errs() << "  Created hyperblock with "
                 << hyperblock_op.getBody().front().getOperations().size()
                 << " operations\n";

    assert(hyperblock_op && "Hyperblock creation failed");

    // If the loop has results (iter_args), map them to hyperblock results.
    if (top_loop.getNumResults() > 0) {
      llvm::errs() << "  Mapping " << top_loop.getNumResults()
                   << " loop results to hyperblock outputs\n";

      for (size_t i = 0; i < top_loop.getNumResults(); ++i) {
        loop_result_to_hyperblock_result[top_loop.getResult(i)] =
            hyperblock_op.getResult(i);
      }
    }
  }

  // Replaces loop results with hyperblock results BEFORE erasing loops.
  for (auto [loop_result, hb_result] : loop_result_to_hyperblock_result) {
    loop_result.replaceAllUsesWith(hb_result);
  }

  // Step 4: Erases all original loops.
  for (affine::AffineForOp loop : top_level_loops) {
    // Ensures no uses remain.
    assert(loop->use_empty() && "Loop still has uses before erasing");
    loop->erase();
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
    return "Constructs hyperblocks from taskflow tasks by detecting perfect "
           "nested loop bands.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<mlir::taskflow::TaskflowDialect, affine::AffineDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();

    // Walks through all TaskflowTaskOp in the function.
    func_op.walk([&](TaskflowTaskOp task_op) {
      if (failed(transformTask(task_op))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::taskflow::createConstructHyperblockFromTaskPass() {
  return std::make_unique<ConstructHyperblockFromTaskPass>();
}