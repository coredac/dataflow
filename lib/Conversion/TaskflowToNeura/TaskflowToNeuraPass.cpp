#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {
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

  StringRef getArgument() const override { return "convert-taskflow-to-neura"; }
  StringRef getDescription() const override {
    return "Convert taskflow.hyperblock to neura.kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::taskflow::TaskflowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

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
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertTaskflowToNeuraPass() {
  return std::make_unique<ConvertTaskflowToNeuraPass>();
}