#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define GEN_PASS_DEF_FUSEPATTERN
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

    if (!first) {
      return failure();
    }

    // Only fuses if the first fadd is not reused elsewhere.
    if (!first->hasOneUse()) {
      return failure();
    }

    Location loc = second.getLoc();
    Type type = second.getType();

    auto fused = rewriter.create<neura::FAddFAddOp>(loc, type, first.getLhs(),
                                                    first.getRhs(), tail);

    rewriter.replaceOp(second, fused.getResult());
    rewriter.eraseOp(first);
    return success();
  }
};

struct FuseFMulFAddPattern : public OpRewritePattern<neura::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::FAddOp add,
                                PatternRewriter &rewriter) const override {
    Value lhs = add.getLhs();
    Value rhs = add.getRhs();

    neura::FMulOp lhs_op = nullptr;
    neura::FMulOp rhs_op = nullptr;
    if (lhs && rhs) {
      lhs_op = lhs.getDefiningOp<neura::FMulOp>();
      rhs_op = rhs.getDefiningOp<neura::FMulOp>();
    } else if (lhs && !rhs) {
      lhs_op = lhs.getDefiningOp<neura::FMulOp>();
    } else {
      llvm::errs() << "FuseMulAddPattern: both lhs and rhs are null\n";
      return failure();
    }

    neura::FMulOp fmul = nullptr;
    Value other;

    // Case 1: fmul is on the LHS.
    if (lhs_op) {
      fmul = lhs_op;
      other = rhs;
    }
    // Case 2: fmul is on the RHS.
    else if (rhs_op) {
      fmul = rhs_op;
      other = lhs;
    }

    if (!fmul) {
      return failure();
    }

    // Optional: only fuses if fmul has a single use.
    if (!fmul->hasOneUse()) {
      return failure();
    }

    Location loc = add.getLoc();
    Type type = add.getType();

    OperationState state(loc, neura::FMulFAddOp::getOperationName());
    state.addTypes(type);
    state.addOperands({fmul.getLhs(), other});
    if (Value rhs_val = fmul.getRhs()) {
      state.addOperands({rhs_val});
    }

    if (fmul->hasAttr("rhs_const_value")) {
      state.addAttribute("rhs_const_value", fmul->getAttr("rhs_const_value"));
    }

    // auto fused = rewriter.create<neura::FMulFAddOp>(loc, type, fmul.getLhs(),
    //                                                 fmul.getRhs(), other);

    Operation *fused_op = rewriter.create(state);

    rewriter.replaceOp(add, fused_op->getResult(0));
    rewriter.eraseOp(fmul);
    return success();
  }
};

struct FuseMulAddPattern : public OpRewritePattern<neura::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::AddOp add,
                                PatternRewriter &rewriter) const override {
    Value lhs = add.getLhs();
    Value rhs = add.getRhs();

    neura::MulOp lhs_op = nullptr;
    neura::MulOp rhs_op = nullptr;
    if (lhs && rhs) {
      lhs_op = lhs.getDefiningOp<neura::MulOp>();
      rhs_op = rhs.getDefiningOp<neura::MulOp>();
    } else if (lhs && !rhs) {
      lhs_op = lhs.getDefiningOp<neura::MulOp>();
    } else {
      llvm::errs() << "FuseMulAddPattern: both lhs and rhs are null\n";
      return failure();
    }

    neura::MulOp mul = nullptr;
    Value other;

    if (lhs_op) {
      mul = lhs_op;
      other = rhs;
    } else if (rhs_op) {
      mul = rhs_op;
      other = lhs;
    }

    if (!mul) {
      return failure();
    }

    if (!mul->hasOneUse()) {
      return failure();
    }

    Location loc = add.getLoc();
    Type type = add.getType();

    OperationState state(loc, neura::MulAddOp::getOperationName());
    state.addTypes(type);
    state.addOperands({mul.getLhs(), other});

    if (Value rhs_val = mul.getRhs()) {
      state.addOperands({rhs_val});
    }

    if (mul->hasAttr("rhs_const_value")) {
      state.addAttribute("rhs_const_value", mul->getAttr("rhs_const_value"));
    }

    Operation *fused_op = rewriter.create(state);
    rewriter.replaceOp(add, fused_op->getResult(0));
    rewriter.eraseOp(mul);
    return success();
  }
};

struct FusePatternPass
    : public PassWrapper<FusePatternPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusePatternPass)

  StringRef getArgument() const override { return "fuse-pattern"; }
  StringRef getDescription() const override {
    return "Apply Neura fusion patterns.";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseFAddFAddPattern>(&getContext(), 2);
    patterns.add<FuseFMulFAddPattern>(&getContext(), 3);
    patterns.add<FuseMulAddPattern>(&getContext(), 3);
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
std::unique_ptr<Pass> createFusePatternPass() {
  return std::make_unique<FusePatternPass>();
}
} // namespace mlir::neura
