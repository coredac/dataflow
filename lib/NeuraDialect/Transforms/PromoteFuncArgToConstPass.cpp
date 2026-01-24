#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include <cassert>
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_PROMOTEFUNCARGTOCONST
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
/**
 * @brief Specializes a region by "internalizing" its input arguments as
 * constants.
 *
 * This function performs a redirection of the dataflow. It identifies all
 * input arguments of the entry block, creates a corresponding
 * `neura::ConstantOp` for each, and re-links all internal operations to use
 * these constants instead of the original block parameters.
 *
 * ### Example Transformation:
 * * **Before:**
 * @code
 * func.func @compute(%arg0: i32) {
 * %0 = arith.addi %arg0, %arg0 : i32
 * return %0 : i32
 * }
 * @endcode
 * * **After:**
 * @code
 * func.func @compute(%arg0: i32) {
 * %0 = "neura.constant"() {value = "%arg0"} : () -> i32
 * %1 = arith.addi %0, %0 : i32  // Uses replaced
 * return %1 : i32
 * }
 * @endcode
 *
 * @param region The MLIR Region (typically a function body) to transform.
 * @return Success if the transformation was applied (even if the region was
 * empty).
 */
LogicalResult promoteFunctionArgsToConstants(Region &region) {
  if (region.empty()) {
    return success();
  }

  Block &entry_block = region.front();
  OpBuilder builder(&entry_block, entry_block.begin());

  // Collects all function arguments.
  SmallVector<BlockArgument, 4> args(entry_block.getArguments().begin(),
                                     entry_block.getArguments().end());

  // Creates a constant operation for each function argument.
  for (auto [idx, arg] : llvm::enumerate(args)) {
    // For constant operation, no predicate.
    auto const_op = builder.create<neura::ConstantOp>(
        arg.getLoc(), arg.getType(),
        builder.getStringAttr("\%arg" + std::to_string(idx)));
    arg.replaceAllUsesWith(const_op.getResult());
  }

  return success();
}

struct PromoteFuncArgToConstPass
    : public PassWrapper<PromoteFuncArgToConstPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteFuncArgToConstPass)

  StringRef getArgument() const override { return "promote-func-arg-to-const"; }
  StringRef getDescription() const override {
    return "Promotes function arguments to constants.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    module_op.walk([&](Operation *op) {
      Region *region = nullptr;
      if (auto func_op = dyn_cast<func::FuncOp>(op)) {
        auto accel_attr =
            func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
        if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
          return;
        }
        region = &func_op.getBody();
      } else if (auto llvm_func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        auto accel_attr =
            llvm_func->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
        if (!accel_attr || accel_attr.getValue() != accel::kNeuraTarget) {
          return;
        }
        region = &llvm_func.getBody();
      } else {
        return;
      }

      if (!region || region->empty()) {
        return;
      }

      if (failed(promoteFunctionArgsToConstants(*region))) {
        signalPassFailure();
        return;
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createPromoteFuncArgToConstPass() {
  return std::make_unique<PromoteFuncArgToConstPass>();
}
} // namespace mlir::neura