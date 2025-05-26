#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Conversion/ConversionPasses.h"

namespace mlir {
namespace neura {
// Uses arith2neura instead of llvm to avoid conflicts.
namespace arith2neura {

#include "ArithToNeuraPatterns.inc"

} // namespace arith2neura
} // namespace neura
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace mlir::neura;

#define GEN_PASS_DEF_LOWERARITHTONEURA
#include "NeuraDialect/NeuraPasses.h.inc"

namespace{

struct LowerArithToNeuraPass
    : public PassWrapper<LowerArithToNeuraPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerArithToNeuraPass)

  StringRef getArgument() const override { return "lower-arith-to-neura"; }
  StringRef getDescription() const override {
    return "Lower arith dialect operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::neura::arith2neura::populateWithGenerated(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerArithToNeuraPass() {
  return std::make_unique<LowerArithToNeuraPass>();
}
