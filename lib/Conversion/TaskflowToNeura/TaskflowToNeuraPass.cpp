#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
//==============================================================================
// innermost Mode.
//==============================================================================
// Checks if an affine.for loop is innermost (has no nested loops).
static bool isInnermostLoop(affine::AffineForOp for_op) {
  bool has_nested_loops = false;
  for_op.getBody()->walk([&](affine::AffineForOp) { has_nested_loops = true; });
  return !has_nested_loops;
}

// Wraps an innermost affine.for loop in a neura.kernel operation.
static LogicalResult wrapInnermostLoopAsKernel(affine::AffineForOp for_op,
                                               OpBuilder &builder,
                                               unsigned &kernel_id) {
  Location loc = for_op.getLoc();

  // Collects values that need to be captured by the kernel.
  llvm::SetVector<Value> captured_values;
  getUsedValuesDefinedAbove(for_op.getRegion(), captured_values);

  // Checks if the loop has output values.
  bool has_outputs = !for_op.getResults().empty();

  // Creates the neura.kernel operation.
  builder.setInsertionPoint(for_op);
  SmallVector<Value> inputs(captured_values.begin(), captured_values.end());

  neura::KernelOp kernel_op = builder.create<neura::KernelOp>(
      loc, /*output_types=*/for_op->getResultTypes(),
      /*inputs=*/inputs,
      /*iter_args_init=*/ValueRange{},
      /*cgra_id*/ nullptr,
      /*kernel_name*/ nullptr,
      /*accelerator*/ nullptr);

  // Sets kernel name.
  std::string kernel_name = "kernel_" + std::to_string(kernel_id++);
  kernel_op.setKernelNameAttr(builder.getStringAttr(kernel_name));

  // Creates the kernel body block.
  Block *kernel_body = new Block();
  kernel_op.getBody().push_back(kernel_body);

  // Adds block arguments for captured values.
  IRMapping mapping;
  for (Value captured : captured_values) {
    BlockArgument arg = kernel_body->addArgument(captured.getType(), loc);
    mapping.map(captured, arg);
  }

  // Clones the loop into the kernel body.
  builder.setInsertionPointToStart(kernel_body);
  Operation *cloned_loop = builder.clone(*for_op, mapping);

  // Adds yield operation.
  builder.setInsertionPointToEnd(kernel_body);
  if (has_outputs) {
    SmallVector<Value> yield_operands(cloned_loop->getResults());
    builder.create<neura::YieldOp>(loc, ValueRange{}, yield_operands);
  } else {
    builder.create<neura::YieldOp>(loc);
  }

  // Replaces uses of the original loop's results with kernel results.
  if (has_outputs) {
    for (auto [orig_result, kernel_result] :
         llvm::zip(for_op->getResults(), kernel_op.getResults())) {
      orig_result.replaceAllUsesWith(kernel_result);
    }
  }

  // Erases the original loop.
  for_op.erase();

  return success();
}

//===============================================================================
// hyperblock Mode.
//===============================================================================
// Pattern to convert taskflow.hyperblock to neura.kernel.
//
// Hyperblock structure:
//   %result = taskflow.hyperblock(%idx, %iter_init) {
//   ^bb0(%idx_arg: index, %iter_arg: T):
//     ... body ...
//     taskflow.hyperblock.yield outputs(%next_iter : T)
//   } : (index, T) -> T
//
// Kernel structure:
//   %result = neura.kernel ins(%idx, %live_in...) iter_args(%iter_init) {
//   ^bb0(%idx_arg: index, %live_in_args..., %iter_arg: T):
//     ... body ...
//     neura.yield iter_args(%next_iter) results(%next_iter)
//   } -> T
//
// Block argument order must match:
//   Hyperblock: [indices..., iter_args...]
//   Kernel:     [inputs (indices + live_ins)..., iter_args...]
struct HyperblockToKernelPattern
    : public OpRewritePattern<TaskflowHyperblockOp> {
  using OpRewritePattern<TaskflowHyperblockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowHyperblockOp hyperblock_op,
                                PatternRewriter &rewriter) const override {
    Location loc = hyperblock_op.getLoc();

    // Finds the parent task to access task's block arguments.
    TaskflowTaskOp task_op = hyperblock_op->getParentOfType<TaskflowTaskOp>();
    if (!task_op) {
      return failure();
    }

    // Asserts that each task contains only one hyperblock.
    int hyperblock_count = 0;
    task_op.walk([&](TaskflowHyperblockOp op) { hyperblock_count++; });
    assert(hyperblock_count == 1 &&
           "Each taskflow.task should contain only one hyperblock");

    Block &hb_block = hyperblock_op.getBody().front();
    Block &task_block = task_op.getBody().front();

    // Gets hyperblock operands.
    SmallVector<Value> indices(hyperblock_op.getIndices());
    DenseSet<Value> indices_set(indices.begin(), indices.end());
    SmallVector<Value> iter_args_init(hyperblock_op.getIterArgs());
    DenseSet<Value> iter_args_init_set(iter_args_init.begin(),
                                       iter_args_init.end());
    size_t num_indices = indices.size();
    size_t num_iter_args_init = iter_args_init.size();

    // Collects live-in values of the hyperblock: task block arguments used in
    // the hyperblock body.
    llvm::DenseSet<Value> live_in_set;
    SmallVector<Value> live_in_values;

    hyperblock_op.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == &task_block) {
            if (iter_args_init_set.contains(operand) ||
                indices_set.contains(operand)) {
              // Skips iter args and indices.
              continue;
            }
            if (live_in_set.insert(operand).second) {
              live_in_values.push_back(operand);
            }
          } else {
            assert(blockArg.getOwner() == &hb_block &&
                   "Unexpected block argument from other block");
          }
        } else if (operand.getDefiningOp()) {
          Operation *def_op = operand.getDefiningOp();
          llvm::errs() << "[taskflow2neura] Operand from op: "
                       << *(operand.getDefiningOp()) << "\n";
          assert(((isa<TaskflowCounterOp>(def_op) &&
                   def_op->getParentOp() == task_op) ||
                  (hyperblock_op->isProperAncestor(def_op))) &&
                 "Unexpected non-block-arg operand in hyperblock");
        }
      }
    });

    // Builds the neura.kernel inputs: [indices..., live_ins...].
    SmallVector<Value> kernel_inputs;
    kernel_inputs.append(indices);
    kernel_inputs.append(live_in_values);

    // Result types from hyperblock.
    SmallVector<Type> resultTypes(hyperblock_op.getResultTypes());

    // Creates neura.kernel.
    neura::KernelOp kernelOp = rewriter.create<neura::KernelOp>(
        loc, resultTypes, kernel_inputs, iter_args_init,
        /*Optional cgra_id*/ nullptr, /*Optional kernel_name*/ nullptr,
        /*Optional accelerator*/ nullptr);

    // Creates the entry block for kernel.
    Region &kernel_region = kernelOp.getBody();
    Block *entry_block = rewriter.createBlock(&kernel_region);

    IRMapping mapping;

    // Kernel block argument layout: [inputs..., iter_args...]
    // Where inputs = [indices..., live_ins...]
    //
    // Hyperblock block argument layout: [indices..., iter_args...]

    // 1. Adds block arguments for indices and map to hyperblock's index args.
    for (size_t i = 0; i < num_indices; ++i) {
      BlockArgument kernel_indices_arg =
          entry_block->addArgument(indices[i].getType(), loc);
      BlockArgument hb_arg = hb_block.getArgument(i);
      mapping.map(hb_arg, kernel_indices_arg);
    }

    // 2. Adds block arguments for live-in values and map to task block args.
    for (Value live_in : live_in_values) {
      BlockArgument kernel_live_in_arg =
          entry_block->addArgument(live_in.getType(), loc);
      mapping.map(live_in, kernel_live_in_arg);
    }

    // 3. Adds block arguments for iter_args and map to hyperblock's iter_args.
    for (size_t i = 0; i < num_iter_args_init; ++i) {
      BlockArgument kernel_iter_arg =
          entry_block->addArgument(iter_args_init[i].getType(), loc);
      BlockArgument hb_arg = hb_block.getArgument(num_indices + i);
      mapping.map(hb_arg, kernel_iter_arg);
    }

    // Clones hyperblock body into kernel.
    rewriter.setInsertionPointToEnd(entry_block);
    for (Operation &op : hb_block.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Converts hyperblock.yield to neura.yield.
    TaskflowHyperblockYieldOp hb_yield_op =
        cast<TaskflowHyperblockYieldOp>(hb_block.getTerminator());

    SmallVector<Value> iter_args_next;
    SmallVector<Value> results;

    // Maps yield outputs.
    for (Value out : hb_yield_op.getResults()) {
      Value mapped = mapping.lookupOrDefault(out);
      results.push_back(mapped);
    }

    for (Value iter_arg : hb_yield_op.getIterArgsNext()) {
      Value mapped = mapping.lookupOrDefault(iter_arg);
      iter_args_next.push_back(mapped);
    }

    rewriter.create<neura::YieldOp>(loc, iter_args_next, results);

    // Replaces hyperblock with kernel.
    rewriter.replaceOp(hyperblock_op, kernelOp.getResults());

    return success();
  }
};

struct InternalizeCounterPattern : public OpRewritePattern<neura::KernelOp> {
  using OpRewritePattern<neura::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::KernelOp kernel_op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> inputs(kernel_op.getInputs());
    SmallVector<Value> iter_args_init(kernel_op.getIterArgsInit());

    // Finds counter inputs: inputs defined by taskflow.counter ops.
    SmallVector<std::pair<size_t, TaskflowCounterOp>> counter_inputs;

    for (size_t i = 0; i < inputs.size(); i++) {
      if (TaskflowCounterOp counter_op =
              inputs[i].getDefiningOp<TaskflowCounterOp>()) {
        counter_inputs.push_back({i, counter_op});
      }
    }

    // If there is no counter inputs, nothing to do.
    if (counter_inputs.empty()) {
      return failure();
    }

    Location loc = kernel_op.getLoc();
    Block &old_block = kernel_op.getBody().front();

    // Builds new inputs (excluding counter inputs).
    DenseSet<size_t> counter_idx_set;
    for (auto &[idx, _] : counter_inputs) {
      counter_idx_set.insert(idx);
    }
    SmallVector<Value> new_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (!counter_idx_set.contains(i)) {
        new_inputs.push_back(inputs[i]);
      }
    }

    // Creates new kernel with updated inputs.
    SmallVector<Type> result_types(kernel_op.getResultTypes());
    neura::KernelOp new_kernel_op = rewriter.create<neura::KernelOp>(
        loc, result_types, new_inputs, iter_args_init,
        /*cgra_id=*/kernel_op.getCgraIdAttr(),
        /*kernel_name=*/kernel_op.getKernelNameAttr(),
        /*accelerator=*/kernel_op.getAcceleratorAttr());

    // Creates the entry block for new kernel.
    Region &new_region = new_kernel_op.getBody();
    Block *new_block = rewriter.createBlock(&new_region);

    IRMapping mapping;
    // Maps non-counter input block arguments.
    for (size_t i = 0; i < inputs.size(); i++) {
      BlockArgument old_arg = old_block.getArgument(i);
      if (!counter_idx_set.contains(i)) {
        BlockArgument new_arg = new_block->addArgument(old_arg.getType(), loc);
        mapping.map(old_arg, new_arg);
      }
    }

    // Maps iter_args block arguments.
    size_t num_inputs = inputs.size();
    for (size_t i = 0; i < iter_args_init.size(); i++) {
      BlockArgument old_arg = old_block.getArgument(num_inputs + i);
      BlockArgument new_arg = new_block->addArgument(old_arg.getType(), loc);
      mapping.map(old_arg, new_arg);
    }

    // Inserts neura.counter ops at the start of the new block.
    rewriter.setInsertionPointToStart(new_block);
    for (auto &[old_idx, source_counter] : counter_inputs) {
      BlockArgument old_counter_arg = old_block.getArgument(old_idx);

      // Creates neura.counter op.
      neura::CounterOp new_counter_op = rewriter.create<neura::CounterOp>(
          source_counter.getLoc(), old_counter_arg.getType(),
          source_counter.getLowerBoundAttr(),
          source_counter.getUpperBoundAttr(), source_counter.getStepAttr(),
          source_counter.getCounterTypeAttr(),
          source_counter.getCounterIdAttr());
      mapping.map(old_counter_arg, new_counter_op.getCurrentIndex());
    }

    // Clones rest of the body.
    rewriter.setInsertionPointToEnd(new_block);
    for (Operation &op : old_block.getOperations()) {
      rewriter.clone(op, mapping);
    }

    // Replaces old kernel with new kernel.
    rewriter.replaceOp(kernel_op, new_kernel_op.getResults());

    return success();
  }
};

struct ConvertTaskflowToNeuraPass
    : public PassWrapper<ConvertTaskflowToNeuraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTaskflowToNeuraPass)

  // Adds default constructor and copy constructor.
  ConvertTaskflowToNeuraPass() = default;
  ConvertTaskflowToNeuraPass(const ConvertTaskflowToNeuraPass &pass)
      : PassWrapper<ConvertTaskflowToNeuraPass, OperationPass<ModuleOp>>(pass) {
  }

  // Pass option to control conversion mode.
  Option<std::string> conversionMode{
      *this, "mode",
      llvm::cl::desc("Conversion mode: 'hyperblock' or 'innermost'"),
      llvm::cl::init("hyperblock")};

  StringRef getArgument() const override { return "convert-taskflow-to-neura"; }
  StringRef getDescription() const override {
    return "Convert taskflow operations to neura.kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::taskflow::TaskflowDialect>();
    registry.insert<mlir::affine::AffineDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Validates conversion mode.
    if (conversionMode != "hyperblock" && conversionMode != "innermost") {
      module.emitError("Invalid conversion mode: ")
          << conversionMode << ". Must be 'hyperblock' or 'innermost'.";
      signalPassFailure();
      return;
    }

    if (conversionMode == "innermost") {
      // Mode: innermost - Wraps only innermost loops as neura.kernel in each
      // task.

      SmallVector<TaskflowTaskOp> task_ops;
      module.walk([&](TaskflowTaskOp task_op) { task_ops.push_back(task_op); });

      OpBuilder builder(ctx);
      unsigned kernel_id = 0;

      for (TaskflowTaskOp task_op : task_ops) {
        // Collects all innermost affine.for loops in the task.
        SmallVector<affine::AffineForOp> innermost_loops;
        task_op.walk([&](affine::AffineForOp for_op) {
          if (isInnermostLoop(for_op)) {
            innermost_loops.push_back(for_op);
          }
        });

        // Wraps each innermost affine.for loop in a neura.kernel operation.
        for (affine::AffineForOp for_op : innermost_loops) {
          if (failed(wrapInnermostLoopAsKernel(for_op, builder, kernel_id))) {
            signalPassFailure();
            return;
          }
        }
      }
    } else {
      // Mode: hyperblock - Converts entire hyperblock to neura.kernel.
      // Phase 1: Converts hyperblocks to kernels.
      {
        RewritePatternSet patterns(ctx);
        patterns.add<HyperblockToKernelPattern>(ctx);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
          signalPassFailure();
          return;
        }
      }

      // Phase 2: Internalizes counters into kernels.
      {
        RewritePatternSet patterns(ctx);
        patterns.add<InternalizeCounterPattern>(ctx);

        if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertTaskflowToNeuraPass() {
  return std::make_unique<ConvertTaskflowToNeuraPass>();
}