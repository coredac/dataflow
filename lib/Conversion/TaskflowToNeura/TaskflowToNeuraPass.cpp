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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

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
    SmallVector<Value> iter_args_init(hyperblock_op.getIterArgs());
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
            if (live_in_set.insert(operand).second) {
              live_in_values.push_back(operand);
            }
          }
        }
        assert(!operand.getDefiningOp() && "Unexpected non-block-arg operand");
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
        /*cgra_id=*/rewriter.getI32IntegerAttr(0),
        /*kernel_name=*/rewriter.getStringAttr("kernel"),
        /*accelerator=*/rewriter.getStringAttr("neura"));

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
    RewritePatternSet patterns(ctx);
    patterns.add<HyperblockToKernelPattern>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertTaskflowToNeuraPass() {
  return std::make_unique<ConvertTaskflowToNeuraPass>();
}