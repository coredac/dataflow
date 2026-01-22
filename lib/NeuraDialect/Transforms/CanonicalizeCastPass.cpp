#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZECAST
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

LogicalResult canonicalizeCast(Region &region) {
  // Handles block arguments.
  for (Block &block : region.getBlocks()) {
    for (BlockArgument arg : block.getArguments()) {
      if (arg.getType().isIndex()) {
        // Replaces index type with i64.
        arg.setType(IntegerType::get(arg.getContext(), 64));
      }
    }
  }

  region.walk([&](Operation *op) {
    // Handles the value attributes in neura::ConstantOp.
    if (isa<neura::ConstantOp>(op)) {
      Attribute value_attr = op->getAttr("value");
      if (!value_attr) {
        return;
      }
      if (IntegerAttr int_attr = dyn_cast<IntegerAttr>(value_attr)) {
        if (isa<IntegerType>(op->getResult(0).getType())) {
          return;
        }
        if (isa<IndexType>(op->getResult(0).getType())) {
          IntegerAttr new_attr = IntegerAttr::get(
              IntegerType::get(op->getContext(), 64), int_attr.getInt());
          op->setAttr("value", new_attr);
        }
      }
    }

    // Replaces all index types with i64.
    for (OpResult result : op->getOpResults()) {
      auto type = result.getType();
      if (isa<IndexType>(type)) {
        result.setType(mlir::IntegerType::get(op->getContext(), 64));
      }
    }

    if (neura::CastOp cast_op = dyn_cast<neura::CastOp>(op)) {
      StringAttr cast_type_attr =
          cast_op->getAttrOfType<StringAttr>("cast_type");
      if (!cast_type_attr) {
        return;
      }
      StringRef cast_type = cast_type_attr.getValue();

      Type src_type = cast_op->getOperand(0).getType();
      Type dst_type = cast_op->getResult(0).getType();

      // Reomoves the index->i64 or i64->index cast operations.
      if ((cast_type == "index_to_int" && isa<IntegerType>(src_type) &&
           isa<IntegerType>(dst_type) &&
           dyn_cast<IntegerType>(src_type).getWidth() == 64 &&
           dyn_cast<IntegerType>(dst_type).getWidth() == 64) ||
          (cast_type == "int_to_index" && isa<IntegerType>(src_type) &&
           isa<IntegerType>(dst_type) &&
           dyn_cast<IntegerType>(src_type).getWidth() == 64 &&
           dyn_cast<IntegerType>(dst_type).getWidth() == 64)) {
        cast_op->getResult(0).replaceAllUsesWith(cast_op->getOperand(0));
        cast_op->erase();
        return;
      }

      // Changes index->i32 or i32->index casts to i64->i32 or i32->i64.
      if (cast_type == "index_to_int" && isa<IntegerType>(dst_type) &&
          dyn_cast<IntegerType>(dst_type).getWidth() == 32) {
        cast_op->setAttr("cast_type",
                         StringAttr::get(op->getContext(), "i64_to_i32"));
        return;
      }
      if (cast_type == "int_to_index" && isa<IntegerType>(src_type) &&
          dyn_cast<IntegerType>(src_type).getWidth() == 32) {
        cast_op->setAttr("cast_type",
                         StringAttr::get(op->getContext(), "i32_to_i64"));
        return;
      }
      // TODO: Handles other cast types if needed.
    }
  });
  return success();
}

struct CanonicalizeCastPass
    : public PassWrapper<CanonicalizeCastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeCastPass)
  StringRef getArgument() const override { return "canonicalize-cast"; }
  StringRef getDescription() const override {
    return "Canonicalizes cast operations in the Neura dialect, specifically "
           "removing unnecessary index to i64 casts and vice versa.";
  }

  void runOnOperation() override {
    auto module_op = getOperation();

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

      if (failed(canonicalizeCast(*region))) {
        signalPassFailure();
        return;
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createCanonicalizeCastPass() {
  return std::make_unique<CanonicalizeCastPass>();
}
} // namespace mlir::neura