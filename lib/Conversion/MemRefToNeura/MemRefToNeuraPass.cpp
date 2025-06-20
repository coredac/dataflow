#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::neura;

namespace {

struct MemRefLoadLowering : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load_op,
                                PatternRewriter &rewriter) const override {
    // Creates a Neura LoadIndexedOp from the MemRef LoadOp.
    Type result_type = load_op.getType();
    Value memref = load_op.getMemRef();
    ValueRange indices = load_op.getIndices();
    // Optiional predicate: default to null
    rewriter.replaceOpWithNewOp<neura::LoadIndexedOp>(load_op, result_type,
                                                      memref, indices, nullptr);
    return success();
  }
};

struct MemRefStoreLowering : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store_op,
                                PatternRewriter &rewriter) const override {
    // Creates a Neura StoreIndexedOp from the MemRef StoreOp.
    Value value = store_op.getValueToStore();
    Value memref = store_op.getMemRef();
    ValueRange indices = store_op.getIndices();
    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::StoreIndexedOp>(store_op, value, memref,
                                                       indices, nullptr);
    return success();
  }
};

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
    ModuleOp module_op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefLoadLowering>(context);
    patterns.add<MemRefStoreLowering>(context);
    module_op.walk([&](func::FuncOp func_op) {
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target =
            func_op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (target && target.getValue() == mlir::accel::kNeuraTarget) {
          if (failed(applyPatternsGreedily(func_op, std::move(patterns)))) {
            return signalPassFailure();
          }
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerMemRefToNeuraPass() {
  return std::make_unique<LowerMemRefToNeuraPass>();
}
