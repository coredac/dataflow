//===- ConvertCopyToAffineLoopsPass.cpp - Converts memref.copy to loops ---===//
//
// This pass converts memref.copy operations into explicit affine loop nests.
//
//===----------------------------------------------------------------------===//

#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
// Converts memref.copy to nested affine loops with affine.load/store.
struct CopyOpLoweringPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    // Checks if the target has any users besides this copy.
    // If the target (e.g., a subview) is only used by this copy and nothing
    // else, this copy is dead code and should be removed without conversion.
    Value target = copy.getTarget();
    bool has_other_users = false;
    for (auto *user : target.getUsers()) {
      if (user != copy.getOperation()) {
        has_other_users = true;
        break;
      }
    }

    if (!has_other_users) {
      // Target has no users besides this copy, so just erase the copy.
      rewriter.eraseOp(copy);
      return success();
    }

    // Target has other users, convert copy to affine loops.
    rewriter.setInsertionPoint(copy);
    auto loc = copy.getLoc();
    MemRefType memref_type = dyn_cast<MemRefType>(copy.getSource().getType());

    // Creates explicit memory copy using an affine loop nest.
    SmallVector<Value> ivs;
    for (auto dim_size : memref_type.getShape()) {
      auto loop = rewriter.create<affine::AffineForOp>(loc, 0, dim_size);
      rewriter.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Creates affine load from source and store to target.
    Value value =
        rewriter.create<affine::AffineLoadOp>(loc, copy.getSource(), ivs);
    rewriter.create<affine::AffineStoreOp>(loc, value, copy.getTarget(), ivs);

    rewriter.eraseOp(copy);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCopyToAffineLoopsPass
    : public PassWrapper<ConvertCopyToAffineLoopsPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCopyToAffineLoopsPass)

  StringRef getArgument() const final { return "convert-copy-to-affine-loops"; }

  StringRef getDescription() const final {
    return "Convert memref.copy to explicit affine loop nests";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CopyOpLoweringPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertCopyToAffineLoopsPass() {
  return std::make_unique<ConvertCopyToAffineLoopsPass>();
}
