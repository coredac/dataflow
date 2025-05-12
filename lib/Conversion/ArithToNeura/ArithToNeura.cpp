#include "Conversion/ArithToNeura/ArithToNeura.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
struct ArithAddFOpLowering : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
llvm::errs() << "[cheng] step into matchAndRewriter()";
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, op.getType(), op.getLhs(), op.getRhs());

llvm::errs() << "[cheng] Matched arith.addf: ";
// op.dump();

    return success();
  }
};

struct LowerArithToNeuraPass
    : public PassWrapper<LowerArithToNeuraPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerArithToNeuraPass)

  StringRef getArgument() const override { return "lower-arith-to-neura"; }
  StringRef getDescription() const override {
    return "Lower arithmetic operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    // getContext().loadDialect<mlir::neura::NeuraDialect>();

    RewritePatternSet patterns(&getContext());
    llvm::errs() << "[cheng] check runOnOperation: ";
    getOperation().dump();
    getOperation().walk([](Operation *op) {
      llvm::errs() << "[cheng] Saw op: " << op->getName() << "\n";
    });
    patterns.add<ArithAddFOpLowering>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createLowerArithToNeuraPass() {
  return std::make_unique<LowerArithToNeuraPass>();
}
