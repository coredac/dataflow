#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "NeuraDialect/NeuraOps.h"

using namespace mlir;

namespace {

struct FuseFAddFAddPattern : public RewritePattern {
  FuseFAddFAddPattern(MLIRContext *ctx)
      : RewritePattern("neura.fadd", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto first = dyn_cast<neura::FAddOp>(op);
    if (!first || !first->hasOneUse()) return failure();

    auto user = dyn_cast<neura::FAddOp>(*first->getUsers().begin());
    if (!user) return failure();

    Location loc = user.getLoc();
    Type type = user.getType();

    auto fused = rewriter.create<neura::FAddFAddOp>(loc, type,
      first.getLhs(), first.getRhs(), user.getRhs());

    rewriter.replaceOp(user, fused.getResult());
    rewriter.eraseOp(first);
    return success();
  }
};

struct FusePatternsPass : public PassWrapper<FusePatternsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePatternsPass)

  StringRef getArgument() const override { return "fuse-patterns"; }
  StringRef getDescription() const override { return "Apply Neura fusion patterns."; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseFAddFAddPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFusePatternsPass() {
  return std::make_unique<FusePatternsPass>();
}
} // namespace mlir::neura

