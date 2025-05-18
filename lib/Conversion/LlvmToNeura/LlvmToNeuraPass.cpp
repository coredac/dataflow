#include "Conversion/LlvmToNeura/LlvmToNeura.h"
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
    patterns.add<LlvmAddToNeuraAdd>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();
    for (Operation &func_op : module_op.getOps()) {
      if (LLVM::LLVMFuncOp llvm_func_op = dyn_cast<LLVM::LLVMFuncOp>(&func_op)) {
        if (failed(applyPatternsAndFoldGreedily(llvm_func_op.getBody(), frozen)))
          signalPassFailure();
      } else if (func::FuncOp mlir_func_op = dyn_cast<func::FuncOp>(&func_op)) {
        if (failed(applyPatternsAndFoldGreedily(mlir_func_op.getBody(), frozen)))
          signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createLowerLlvmToNeuraPass() {
  return std::make_unique<LowerLlvmToNeuraPass>();
}
