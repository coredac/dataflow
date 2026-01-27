#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

#define GEN_PASS_DEF_ASSIGNACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// Checks if a function contains any neura.kernel operations.
static bool containsNeuraKernelOp(FunctionOpInterface func_op) {
  bool has_kernel = false;
  func_op.walk([&](neura::KernelOp kernel_op) {
    has_kernel = true;
    return WalkResult::interrupt();
  });
  return has_kernel;
}

struct AssignAcceleratorPass
    : public PassWrapper<AssignAcceleratorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignAcceleratorPass)

  StringRef getArgument() const override { return "assign-accelerator"; }
  StringRef getDescription() const override {
    return "Tags non-main functions as neura.kernel.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());

    // Firstly assigns accelerator to all neura.kernel ops.
    module.walk([&](neura::KernelOp kernel_op) {
      // Handles neura.kernel ops.
      if (!kernel_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        kernel_op->setAttr(mlir::accel::kAcceleratorAttr,
                           builder.getStringAttr(mlir::accel::kNeuraTarget));
      }
    });

    // Secondly assigns accelerator to functions.
    // Skips functions that:
    //   1. Are named "main";
    //   2. Already have accelerator attribute;
    //   3. Contain neura.kernel operations.
    module.walk([&](Operation *op) {
      if (auto func = dyn_cast<FunctionOpInterface>(op)) {
        if (func.getName() != "main" && !func.isExternal() &&
            !func->hasAttr(mlir::accel::kAcceleratorAttr) &&
            !containsNeuraKernelOp(func)) {
          func->setAttr(mlir::accel::kAcceleratorAttr,
                        builder.getStringAttr(mlir::accel::kNeuraTarget));
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
