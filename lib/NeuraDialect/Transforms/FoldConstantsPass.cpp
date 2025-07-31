#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_FOLDCONSTANT
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct FuseConstantAndGrantPattern
    : public OpRewritePattern<neura::ConstantOp> {
  using OpRewritePattern<neura::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::ConstantOp constant_op,
                                PatternRewriter &rewriter) const override {
    bool made_change = false;

    // Checks if the constant operation is used by a grant_once or grant_always
    // operation.
    for (auto user : constant_op->getUsers()) {
      llvm::errs() << "Checking use: " << *user << "\n";
      if (isa<neura::GrantOnceOp>(user) || isa<neura::GrantAlwaysOp>(user)) {
        if (neura::GrantOnceOp grant_once_op =
                dyn_cast<neura::GrantOnceOp>(user)) {
          auto new_grant_once_op = rewriter.create<neura::GrantOnceOp>(
              grant_once_op.getLoc(), grant_once_op.getResult().getType(),
              /*value=*/nullptr, constant_op->getAttr("value"));
          // Replaces the original constant operation with the new one.
          rewriter.replaceOp(grant_once_op, new_grant_once_op);
          made_change = true;
        } else if (neura::GrantAlwaysOp grant_always_op =
                       dyn_cast<neura::GrantAlwaysOp>(user)) {
          auto new_grant_always_op = rewriter.create<neura::GrantAlwaysOp>(
              grant_always_op.getLoc(), grant_always_op.getResult().getType(),
              /*value=*/nullptr, constant_op->getAttr("value"));
          // Replaces the original constant operation with the new one.
          rewriter.replaceOp(grant_always_op, new_grant_always_op);
          made_change = true;
        }
      }
    }

    if (constant_op->use_empty()) {
      // If the constant operation has no users, it can be removed.
      rewriter.eraseOp(constant_op);
      made_change = true;
    }

    return success(made_change);
  }
};

struct FoldConstantsPass
    : public PassWrapper<FoldConstantsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConstantsPass)

  StringRef getArgument() const override { return "fold-constants"; }
  StringRef getDescription() const override {
    return "Fold constant operations.";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConstantAndGrantPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

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
std::unique_ptr<Pass> createFoldConstantsPass() {
  return std::make_unique<FoldConstantsPass>();
}
} // namespace mlir::neura
