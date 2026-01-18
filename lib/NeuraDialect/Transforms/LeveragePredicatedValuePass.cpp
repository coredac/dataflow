#include "Common/AcceleratorAttrs.h"
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

#define GEN_PASS_DEF_LEVERAGEPREDICATEDVALUE
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

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

    // Processes each function.
    module.walk([&](FunctionOpInterface func) {
      auto accel_attr =
          func->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }
      // Converts block argument types to predicated values.
      func.walk([&](Block *block) {
        // skips the entry (first) block of the function.
        if (block == &block->getParent()->front()) {
          return;
        }

        for (BlockArgument arg : block->getArguments()) {
          Type orig_type = arg.getType();

          // Avoid double-wrapping if already predicated
          if (llvm::isa<neura::PredicatedValue>(orig_type)) {
            continue;
          }

          auto predicated_type = neura::PredicatedValue::get(
              func.getContext(), orig_type,
              IntegerType::get(func.getContext(), 1));
          arg.setType(predicated_type);
        }
      });

      // Gets operations in topological order (operands before users).
      SmallVector<Operation *> orderedOps;
      getOperationsInTopologicalOrder(func, orderedOps);

      // Processes each operation in order.
      for (Operation *op : orderedOps) {
        if (failed(applyPredicatedDataType(op))) {
          llvm::errs() << "Failed to convert op to predicated form: " << *op
                       << "\n";
          signalPassFailure();
          return;
        }
      }
    });
  }

private:
  // Gets operations in topological order.
  void getOperationsInTopologicalOrder(FunctionOpInterface func,
                                       SmallVector<Operation *> &ordered) {
    DenseSet<Operation *> visited;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Uses standard DFS to build topological order.
      if (visited.contains(op)) {
        return;
      }

      // Visits operands first.
      for (Value operand : op->getOperands()) {
        if (auto def_op = operand.getDefiningOp()) {
          if (!visited.contains(def_op)) {
            visited.insert(def_op);
            ordered.push_back(def_op);
          }
        }
      }

      // Then visits current op.
      if (!visited.contains(op)) {
        visited.insert(op);
        ordered.push_back(op);
      }
    });
  }

  // Converts a single operation to use predicated values.
  LogicalResult applyPredicatedDataType(Operation *op) {
    // Skips if not a Neura op.
    if (op->getDialect()->getNamespace() != accel::kNeuraTarget) {
      return success();
    }

    // Skips if no results or already predicated.
    if (op->getNumResults() == 0 ||
        llvm::any_of(op->getResultTypes(), [](Type t) {
          return mlir::isa<mlir::neura::PredicatedValue>(t);
        })) {
      return success();
    }

    // Converts result types to predicated form.
    OpBuilder builder(op);
    SmallVector<Type> newResults;
    for (Type t : op->getResultTypes()) {
      auto predicated_type = mlir::neura::PredicatedValue::get(
          op->getContext(), t, builder.getI1Type());
      newResults.push_back(predicated_type);
    }

    // Clones with new result types.
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(op->getOperands());
    state.addTypes(newResults);
    state.addAttributes(op->getAttrs());
    Operation *newOp = builder.create(state);

    // Replaces old op.
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