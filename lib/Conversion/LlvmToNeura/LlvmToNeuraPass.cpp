#include "Conversion/LlvmToNeura/LlvmToNeura.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct LlvmAddFOpLowering : public OpRewritePattern<mlir::LLVM::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FAddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

struct LowerLlvmToNeuraPass
    : public PassWrapper<LowerLlvmToNeuraPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLlvmToNeuraPass)

  StringRef getArgument() const override { return "lower-llvm-to-neura"; }
  StringRef getDescription() const override {
    return "Lower LLVM operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LlvmAddFOpLowering>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createLowerLlvmToNeuraPass() {
  return std::make_unique<LowerLlvmToNeuraPass>();
}
