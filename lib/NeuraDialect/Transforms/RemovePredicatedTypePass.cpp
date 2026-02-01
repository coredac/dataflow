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

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Processes each function.
    module.walk([&](FunctionOpInterface func) {
      auto accel_attr =
          func->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
        return;
      }

      // Converts block arguments.
      func.walk([&](Block *block) {
        // Processes block arguments.
        for (BlockArgument arg : block->getArguments()) {
          Type orig_type = arg.getType();
          if (auto predicated_type =
                  llvm::dyn_cast<neura::PredicatedValue>(orig_type)) {
            arg.setType(predicated_type.getValueType());
          }
        }
      });

      // Gets operations in topological order.
      SmallVector<Operation *> ordered_ops;
      getOperationsInTopologicalOrder(func, ordered_ops);

      // Processes each operation in topological order.
      for (Operation *op : ordered_ops) {
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
  // Gets operations in topological order.
  void getOperationsInTopologicalOrder(FunctionOpInterface func,
                                       SmallVector<Operation *> &ordered_ops) {
    DenseSet<Operation *> visited_ops;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (visited_ops.contains(op)) {
        return;
      }

      // Visits operands first.
      for (Value operand : op->getOperands()) {
        if (Operation *def_op = operand.getDefiningOp()) {
          if (!visited_ops.contains(def_op)) {
            visited_ops.insert(def_op);
            ordered_ops.push_back(def_op);
          }
        }
      }

      if (!visited_ops.contains(op)) {
        visited_ops.insert(op);
        ordered_ops.push_back(op);
      }
    });
  }

  // Converts a single operation from predicated to normal types.
  LogicalResult removePredicatedType(Operation *op) {
    // Skips if not a Neura op.
    if (op->getDialect()->getNamespace() != accel::kNeuraTarget) {
      return success();
    }

    // Skips if no results or no predicated types.
    if (op->getNumResults() == 0 ||
        !llvm::any_of(op->getResultTypes(), [](Type t) {
          return mlir::isa<mlir::neura::PredicatedValue>(t);
        })) {
      return success();
    }

    // Converts result types to non-predicated form.
    OpBuilder builder(op);
    SmallVector<Type> new_results;
    for (Type t : op->getResultTypes()) {
      if (auto predicated_type = llvm::dyn_cast<neura::PredicatedValue>(t)) {
        new_results.push_back(predicated_type.getValueType());
      } else {
        new_results.push_back(t);
      }
    }

    // Creates new operation with updated result types.
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(op->getOperands());
    state.addTypes(new_results);
    state.addAttributes(op->getAttrs());

    // Copies regions if needed.
    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
      state.addRegion();
    }

    Operation *new_op = builder.create(state);

    // Moves regions if any.
    for (unsigned i = 0; i < op->getNumRegions(); ++i) {
      Region &old_region = op->getRegion(i);
      Region &new_region = new_op->getRegion(i);
      new_region.takeBody(old_region);
    }

    // Replaces old op.
    op->replaceAllUsesWith(new_op);
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