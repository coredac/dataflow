#include "Conversion/LlvmToNeura/LlvmToNeura.h"
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

namespace mlir {
namespace neura {
// Uses llvm2neura instead of llvm to avoid conflicts.
namespace llvm2neura {

#include "LlvmToNeuraPatterns.inc"

} // namespace llvm2neura
} // namespace neura
} // namespace mlir

using namespace mlir;

namespace {

// Lowers integer add from mlir.llvm.add to nuera.add. We provide the lowering
// here instead of tablegen due to that mlir.llvm.add uses an EnumProperty
// (IntegerOverflowFlags) defined via MLIR interfaces — which DRR cannot match
// on or extract from.
struct LlvmAddToNeuraAdd : public OpRewritePattern<mlir::LLVM::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};

struct LlvmFMulToNeuraFMul : public OpRewritePattern<mlir::LLVM::FMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVFMulToNeuraVFMul: public OpRewritePattern<mlir::LLVM::FMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xf32>.
    auto vecTy = mlir::dyn_cast<VectorType>(result_type);
    if (!vecTy || !mlir::isa<FloatType>(vecTy.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VFMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LowerLlvmToNeuraPass
    : public PassWrapper<LowerLlvmToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLlvmToNeuraPass)

  StringRef getArgument() const override { return "lower-llvm-to-neura"; }
  StringRef getDescription() const override {
    return "Lower LLVM operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Adds DRR patterns.
    mlir::neura::llvm2neura::populateWithGenerated(patterns);
    patterns.add<LlvmAddToNeuraAdd>(&getContext());
    patterns.add<LlvmFMulToNeuraFMul>(&getContext());
    patterns.add<LlvmVFMulToNeuraVFMul>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](FunctionOpInterface func) {
      if (func->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target = func->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (target && target.getValue() == mlir::accel::kNeuraTarget) {
          for (Region &region : func->getRegions()) {
            if (failed(applyPatternsAndFoldGreedily(region, frozen))) {
              signalPassFailure();
            }
          }
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createLowerLlvmToNeuraPass() {
  return std::make_unique<LowerLlvmToNeuraPass>();
}
