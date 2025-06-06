#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_LeveragePredicatedValue 
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct applyPredicatedDataType : public RewritePattern {
  applyPredicatedDataType(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    llvm::errs() << "Processing op: " << *op << "\n";

    // Skip if not a Neura op or already using predicated values
    if (op->getDialect()->getNamespace() != "neura") {
        llvm::errs() << "Skipping non-Neura op\n";
        return failure();
    }

    if (llvm::any_of(op->getResultTypes(), 
        [](Type t) { return mlir::isa<mlir::neura::PredicatedValue>(t); })) {
        llvm::errs() << "Skipping already predicated op\n";
        return failure();
    }

    // Convert result types to predicated form
    SmallVector<Type> newResults;
    for (Type t : op->getResultTypes()) {
        auto predicatedTy = mlir::neura::PredicatedValue::get(
            op->getContext(),
            t,
            rewriter.getI1Type());
        newResults.push_back(predicatedTy);
    }

    // Clone the operation with new result types
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(op->getOperands());
    state.addTypes(newResults);
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);

    // Replace the old op with the new one
    rewriter.replaceOp(op, newOp->getResults());
    llvm::errs() << "Converted op to predicated form: " << *newOp << "\n";
    if (!newResults.empty()) {
      assert(false);
    }
    return success();
  }
};

struct LeveragePredicatedValuePass
    : public PassWrapper<LeveragePredicatedValuePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LeveragePredicatedValuePass)

  StringRef getArgument() const override { return "leverage-predicated-value"; }
  StringRef getDescription() const override {
    return "Convert values to predicated values in Neura dialect operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Process each function
    module.walk([&](func::FuncOp func) {
      // Get operations in topological order (operands before users)
      SmallVector<Operation*> orderedOps;
      getOperationsInTopologicalOrder(func, orderedOps);

      // Process each operation in order
      for (Operation *op : orderedOps) {
        if (failed(applyPredicatedDataType(op))) {
          llvm::errs() << "Failed to convert op to predicated form: " << *op << "\n";
          signalPassFailure();
          return;
        }
      }
    });
  }

private:
  // Get operations in topological order
  void getOperationsInTopologicalOrder(func::FuncOp func, 
                                     SmallVector<Operation*> &ordered) {
    DenseSet<Operation*> visited;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Use standard DFS to build topological order
      if (visited.contains(op))
        return;
        
      // Visit operands first
      for (Value operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (!visited.contains(defOp)) {
            visited.insert(defOp);
            ordered.push_back(defOp);
          }
        }
      }
      
      // Then visit current op
      if (!visited.contains(op)) {
        visited.insert(op);
        ordered.push_back(op);
      }
    });
  }

  // Convert a single operation to use predicated values
  LogicalResult applyPredicatedDataType(Operation *op) {
    llvm::errs() << "Processing op: " << *op << "\n";

    // Skip if not a Neura op
    if (op->getDialect()->getNamespace() != "neura") {
      llvm::errs() << "Skipping non-Neura op\n";
      return success();
    }

    // Skip if no results or already predicated
    if (op->getNumResults() == 0 || 
        llvm::any_of(op->getResultTypes(), 
          [](Type t) { return mlir::isa<mlir::neura::PredicatedValue>(t); })) {
      return success();
    }

    // Convert result types to predicated form
    OpBuilder builder(op);
    SmallVector<Type> newResults;
    for (Type t : op->getResultTypes()) {
      auto predicatedTy = mlir::neura::PredicatedValue::get(
          op->getContext(),
          t,
          builder.getI1Type());
      newResults.push_back(predicatedTy);
    }

    // Clone with new result types
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(op->getOperands());
    state.addTypes(newResults);
    state.addAttributes(op->getAttrs());
    Operation *newOp = builder.create(state);

    // Replace old op
    op->replaceAllUsesWith(newOp);
    op->erase();
    return success();
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createLeveragePredicatedValuePass() {
  return std::make_unique<LeveragePredicatedValuePass>();
}

} // namespace neura
} // namespace mlir