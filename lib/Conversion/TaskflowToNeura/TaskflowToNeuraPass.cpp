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
struct HyperblockToKernelPattern
    : public OpRewritePattern<TaskflowHyperblockOp> {
  using OpRewritePattern<TaskflowHyperblockOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskflowHyperblockOp hyperblock_op,
                                PatternRewriter &rewriter) const override {
    Location loc = hyperblock_op.getLoc();

    // Find the parent task to get access to task's block arguments.
    auto taskOp = hyperblock_op->getParentOfType<TaskflowTaskOp>();
    if (!taskOp)
      return failure();

    // Collect live-in values: values used in hyperblock but defined outside.
    // These are the task's block arguments that the hyperblock body uses.
    llvm::DenseSet<Value> liveInSet;
    SmallVector<Value> liveInValues;

    Block &hbBlock = hyperblock_op.getBody().front();
    Block &taskBlock = taskOp.getBody().front();

    // Walk hyperblock body to find uses of task block arguments.
    hyperblock_op.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        // Check if operand is a task block argument.
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == &taskBlock) {
            if (liveInSet.insert(operand).second) {
              liveInValues.push_back(operand);
            }
          }
        }
      }
    });

    // Collect iter_args initial values.
    SmallVector<Value> iterArgsInit(hyperblock_op.getIterArgs().begin(),
                                    hyperblock_op.getIterArgs().end());

    // Determine result types.
    SmallVector<Type> resultTypes(hyperblock_op.getResultTypes().begin(),
                                  hyperblock_op.getResultTypes().end());

    // Collect input types.
    SmallVector<Type> inputTypes;
    for (Value v : liveInValues) {
      inputTypes.push_back(v.getType());
    }

    SmallVector<Type> iterArgsTypes;
    for (Value v : iterArgsInit) {
      iterArgsTypes.push_back(v.getType());
    }

    // Create neura.kernel.
    auto kernelOp = rewriter.create<neura::KernelOp>(
        loc, resultTypes, liveInValues, iterArgsInit,
        /*cgra_id=*/rewriter.getI32IntegerAttr(0),
        /*kernel_name=*/rewriter.getStringAttr("kernel"),
        /*accelerator=*/rewriter.getStringAttr("neura"));

    // Create entry block for kernel.
    Region &kernelRegion = kernelOp.getBody();
    Block *entryBlock = rewriter.createBlock(&kernelRegion);

    IRMapping mapping;

    // Add block arguments for live-in values (inputs).
    for (auto [idx, liveIn] : llvm::enumerate(liveInValues)) {
      BlockArgument arg = entryBlock->addArgument(liveIn.getType(), loc);
      mapping.map(liveIn, arg);
    }

    // Add block arguments for iter_args.
    size_t numIndices = hyperblock_op.getIndices().size();
    for (auto [idx, iterArg] : llvm::enumerate(iterArgsInit)) {
      BlockArgument arg = entryBlock->addArgument(iterArg.getType(), loc);
      // Map hyperblock's iter_arg block argument to kernel's block argument.
      mapping.map(hbBlock.getArgument(numIndices + idx), arg);
    }

    // Map hyperblock's index arguments - these will be replaced by counter
    // later. For now, create placeholder block arguments.
    for (size_t i = 0; i < numIndices; ++i) {
      BlockArgument hbArg = hbBlock.getArgument(i);
      BlockArgument arg = entryBlock->addArgument(hbArg.getType(), loc);
      mapping.map(hbArg, arg);
    }

    // Clone hyperblock body into kernel.
    rewriter.setInsertionPointToEnd(entryBlock);
    for (Operation &op : hbBlock.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Convert hyperblock.yield to neura.yield.
    auto yieldOp = cast<TaskflowHyperblockYieldOp>(hbBlock.getTerminator());
    SmallVector<Value> iterArgsNext;
    SmallVector<Value> results;

    for (Value out : yieldOp.getOutputs()) {
      Value mapped = mapping.lookupOrDefault(out);
      // For kernels with iter_args, output goes to both iter_args_next and
      // results.
      iterArgsNext.push_back(mapped);
      results.push_back(mapped);
    }

    rewriter.create<neura::YieldOp>(loc, iterArgsNext, results);

    // Replace hyperblock results with kernel results.
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