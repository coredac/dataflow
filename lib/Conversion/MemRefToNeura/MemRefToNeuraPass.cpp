#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Conversion/ConversionPasses.h"

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_LOWERLLVMTONEURA
#include "NeuraDialect/NeuraPasses.h.inc"


namespace {

struct LowerMemRefToNeuraPass
    : public PassWrapper<LowerMemRefToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMemRefToNeuraPass)

  StringRef getArgument() const override { return "lower-memref-to-neura"; }
  StringRef getDescription() const override {
    return "Lower MemRef operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerMemRefToNeuraPass() {
  return std::make_unique<LowerMemRefToNeuraPass>();
}
