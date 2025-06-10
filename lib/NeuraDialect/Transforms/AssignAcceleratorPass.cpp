#include "Common/AcceleratorAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

#define GEN_PASS_DEF_ASSIGNACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct AssignAcceleratorPass : public PassWrapper<AssignAcceleratorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignAcceleratorPass)

  StringRef getArgument() const override { return "assign-accelerator"; }
  StringRef getDescription() const override { return "Tags non-main functions as neura.kernel."; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());

    module.walk([&](Operation *op) {
      if (auto func = dyn_cast<FunctionOpInterface>(op)) {
        if (func.getName() != "main" &&
            !func.isExternal() &&
            !func->hasAttr(mlir::accel::kAcceleratorAttr)) {
          func->setAttr(mlir::accel::kAcceleratorAttr, builder.getStringAttr(mlir::accel::kNeuraTarget));
        }
      }
    });
  }
};
} // namespace

/// Registers the pass.
namespace mlir {
namespace neura {
std::unique_ptr<Pass> createAssignAcceleratorPass() {
  return std::make_unique<AssignAcceleratorPass>();
}
} // namespace neura
} // namespace mlir

