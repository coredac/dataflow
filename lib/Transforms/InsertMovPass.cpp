#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {
struct InsertMovForNeuraOps : public RewritePattern {
  InsertMovForNeuraOps(MLIRContext *context)
      : RewritePattern(/*matchAnyOpTypeTag=*/MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  // using RewritePattern::RewritePattern;

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() != "neura" ||
        isa<neura::MovOp>(op)) {
      return failure();
    }

    llvm::errs() << "[cheng] step into matching and rewrite";
    Location loc = op->getLoc();

    // Wrap operands in mov
    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      auto mov = rewriter.create<neura::MovOp>(loc, operand.getType(), operand);
      newOperands.push_back(mov);
    }

    // Clone op with new operands
    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);

    // Wrap each result in a mov
    SmallVector<Value> newResults;
    for (Value result : newOp->getResults()) {
      auto mov = rewriter.create<neura::MovOp>(loc, result.getType(), result);
      newResults.push_back(mov);
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct InsertMovPass
    : public PassWrapper<InsertMovPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertMovPass)

  StringRef getArgument() const override { return "insert-mov"; }
  StringRef getDescription() const override {
    return "Insert neura.mov before and after all neura dialect operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertMovForNeuraOps>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createInsertMovPass() {
  return std::make_unique<InsertMovPass>();
}

} // namespace neura
} // namespace mlir
