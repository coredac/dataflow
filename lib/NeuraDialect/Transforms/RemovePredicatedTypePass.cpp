#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define GEN_PASS_DEF_REMOVEPREDICATEDTYPE
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

struct RemovePredicatedTypePass
    : public PassWrapper<RemovePredicatedTypePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemovePredicatedTypePass)

  StringRef getArgument() const override { return "remove-predicated-type"; }
  StringRef getDescription() const override {
    return "Remove predicated types from Neura dialect operations, reverting "
           "to basic types.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each function
    module.walk([&](FunctionOpInterface func) {
      auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
      if (!accel_attr || accel_attr.getValue() != "neura") {
        return;
      }

      // Get operations in topological order
      SmallVector<Operation *> orderedOps;
      getOperationsInTopologicalOrder(func, orderedOps);

      // Process each operation in order
      for (Operation *op : orderedOps) {
        if (failed(removePredicatedType(op))) {
          llvm::errs() << "Failed to convert op from predicated form: " << *op
                       << "\n";
          signalPassFailure();
          return;
        }
      }
    });
  }

private:
  // Get operations in topological order
  void getOperationsInTopologicalOrder(FunctionOpInterface func,
                                       SmallVector<Operation *> &ordered) {
    DenseSet<Operation *> visited;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (visited.contains(op))
        return;

      // Visit operands first
      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (!visited.contains(defOp)) {
            visited.insert(defOp);
            ordered.push_back(defOp);
          }
        }
      }

      if (!visited.contains(op)) {
        visited.insert(op);
        ordered.push_back(op);
      }
    });
  }

  // Convert a single operation from predicated to normal types
  LogicalResult removePredicatedType(Operation *op) {
    // Skip if not a Neura op
    if (op->getDialect()->getNamespace() != "neura")
      return success();

    // Skip if no results or no predicated types
    if (op->getNumResults() == 0 ||
        !llvm::any_of(op->getResultTypes(), [](Type t) {
          return mlir::isa<mlir::neura::PredicatedValue>(t);
        })) {
      return success();
    }

    // Convert result types to non-predicated form
    OpBuilder builder(op);
    SmallVector<Type> newResults;
    for (Type t : op->getResultTypes()) {
      if (auto predicatedType = llvm::dyn_cast<neura::PredicatedValue>(t)) {
        newResults.push_back(predicatedType.getValueType());
      } else {
        newResults.push_back(t);
      }
    }

    // Create new operation with updated result types
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(op->getOperands());
    state.addTypes(newResults);
    state.addAttributes(op->getAttrs());

    // Copy regions if needed
    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
      state.addRegion();
    }

    Operation *newOp = builder.create(state);

    // Move regions if any
    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
      Region &oldRegion = op->getRegion(i);
      Region &newRegion = newOp->getRegion(i);
      newRegion.takeBody(oldRegion);
    }

    // Replace old op
    op->replaceAllUsesWith(newOp);
    op->erase();
    return success();
  }
};

} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createRemovePredicatedTypePass() {
  return std::make_unique<RemovePredicatedTypePass>();
}

} // namespace neura
} // namespace mlir