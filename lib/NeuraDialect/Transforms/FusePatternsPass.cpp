#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "NeuraDialect/NeuraOps.h"

using namespace mlir;

#define GEN_PASS_DEF_FUSEPATTERNS
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

struct FuseFAddFAddPattern : public OpRewritePattern<neura::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FAddOp second,
                                PatternRewriter &rewriter) const override {
    Value lhs = second.getLhs();
    Value rhs = second.getRhs();

    auto lhs_op = lhs.getDefiningOp<neura::FAddOp>();
    auto rhs_op = rhs.getDefiningOp<neura::FAddOp>();

    neura::FAddOp first = nullptr;
    Value tail;

    // Case 1: LHS is another fadd.
    if (lhs_op && second.getRhs()) {
      first = lhs_op;
      tail = second.getRhs();
    }
    // Case 2: RHS is another fadd.
    else if (rhs_op && second.getLhs()) {
      first = rhs_op;
      tail = second.getLhs();
    }

    if (!first)
      return failure();

    // Only fuses if the first fadd is not reused elsewhere.
    if (!first->hasOneUse())
      return failure();

    Location loc = second.getLoc();
    Type type = second.getType();

    auto fused = rewriter.create<neura::FAddFAddOp>(
        loc, type, first.getLhs(), first.getRhs(), tail, Value());

    rewriter.replaceOp(second, fused.getResult());
    rewriter.eraseOp(first);
    return success();
  }
};

struct FuseFMulFAddPattern : public OpRewritePattern<neura::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FAddOp add,
                                PatternRewriter &rewriter) const override {
    auto lhs_op = add.getLhs().getDefiningOp<neura::FMulOp>();
    auto rhs_op = add.getRhs().getDefiningOp<neura::FMulOp>();

    neura::FMulOp fmul = nullptr;
    Value other;

    // Case 1: fmul is on the LHS.
    if (lhs_op && add.getRhs()) {
      fmul = lhs_op;
      other = add.getRhs();
    }
    // Case 2: fmul is on the RHS.
    else if (rhs_op && add.getLhs()) {
      fmul = rhs_op;
      other = add.getLhs();
    }

    if (!fmul)
      return failure();

    // Optional: only fuses if fmul has a single use.
    if (!fmul->hasOneUse())
      return failure();

    Location loc = add.getLoc();
    Type type = add.getType();

    auto fused = rewriter.create<neura::FMulFAddOp>(
        loc, type, fmul.getLhs(), fmul.getRhs(), other, Value());

    rewriter.replaceOp(add, fused.getResult());
    rewriter.eraseOp(fmul);
    return success();
  }
};

struct FusePatternsPass : public PassWrapper<FusePatternsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePatternsPass)

  StringRef getArgument() const override { return "fuse-patterns"; }
  StringRef getDescription() const override { return "Apply Neura fusion patterns."; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseFAddFAddPattern>(&getContext(), 2);
    patterns.add<FuseFMulFAddPattern>(&getContext(), 3);
    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFusePatternsPass() {
  return std::make_unique<FusePatternsPass>();
}
} // namespace mlir::neura

