//===- FoldSubViewPass.cpp - Fold memref.subview into load/store ---------===//
//
// This pass folds memref.subview operations into their affine.load and
// affine.store users by adjusting the access indices. Designed for CGRA
// systems with global addressing.
//
//===----------------------------------------------------------------------===//

#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

// Resolves the source indices for a load/store operation that accesses a
// subview. Computes the adjusted indices to access the source memref directly.
//
// For example:
//   %subview = memref.subview %source[%offset0, %offset1][...][%stride0,
//   %stride1] %val = affine.load %subview[%i, %j]
// becomes:
//   %val = affine.load %source[%i * %stride0 + %offset0, %j * %stride1 +
//   %offset1]
static LogicalResult
resolveSourceIndices(Location loc, PatternRewriter &rewriter,
                     memref::SubViewOp sub_view_op, ValueRange indices,
                     SmallVectorImpl<Value> &source_indices) {
  SmallVector<OpFoldResult> mixed_offsets = sub_view_op.getMixedOffsets();
  SmallVector<OpFoldResult> mixed_sizes = sub_view_op.getMixedSizes();
  SmallVector<OpFoldResult> mixed_strides = sub_view_op.getMixedStrides();

  SmallVector<Value> use_indices;
  // Handles rank-reducing subviews: for every unit-dim size, adds a zero index.
  unsigned result_dim = 0;
  llvm::SmallBitVector unused_dims = sub_view_op.getDroppedDims();
  for (auto dim :
       llvm::seq<unsigned>(0, sub_view_op.getSourceType().getRank())) {
    if (unused_dims.test(dim)) {
      use_indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    } else {
      use_indices.push_back(indices[result_dim++]);
    }
  }

  if (use_indices.size() != mixed_offsets.size()) {
    return failure();
  }

  source_indices.resize(use_indices.size());
  for (auto index : llvm::seq<size_t>(0, mixed_offsets.size())) {
    SmallVector<Value> dynamic_operands;
    AffineExpr expr = rewriter.getAffineDimExpr(0);
    unsigned num_symbols = 0;
    dynamic_operands.push_back(use_indices[index]);

    // Multiplies by stride: index * stride.
    if (auto attr = mixed_strides[index].dyn_cast<Attribute>()) {
      int64_t stride_val = dyn_cast<IntegerAttr>(attr).getInt();
      if (stride_val != 1) {
        expr = expr * stride_val;
      }
    } else {
      dynamic_operands.push_back(dyn_cast<Value>(mixed_strides[index]));
      expr = expr * rewriter.getAffineSymbolExpr(num_symbols++);
    }

    // Adds offset: index * stride + offset.
    if (auto attr = mixed_offsets[index].dyn_cast<Attribute>()) {
      int64_t offset_val = dyn_cast<IntegerAttr>(attr).getInt();
      if (offset_val != 0) {
        expr = expr + offset_val;
      }
    } else {
      dynamic_operands.push_back(dyn_cast<Value>(mixed_offsets[index]));
      expr = expr + rewriter.getAffineSymbolExpr(num_symbols++);
    }

    source_indices[index] = rewriter.create<affine::AffineApplyOp>(
        loc, AffineMap::get(1, num_symbols, expr), dynamic_operands);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {
/// Folds affine.load from a subview into a load from the source memref.
class LoadOpOfSubViewFolder final
    : public OpRewritePattern<affine::AffineLoadOp> {
public:
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto subViewOp = loadOp.getMemRef().getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();

    SmallVector<Value, 4> sourceIndices;
    if (failed(resolveSourceIndices(loadOp.getLoc(), rewriter, subViewOp,
                                    loadOp.getIndices(), sourceIndices)))
      return failure();

    rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
        loadOp, subViewOp.getSource(), sourceIndices);
    return success();
  }
};

/// Folds affine.store to a subview into a store to the source memref.
class StoreOpOfSubViewFolder final
    : public OpRewritePattern<affine::AffineStoreOp> {
public:
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto subViewOp = storeOp.getMemRef().getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();

    SmallVector<Value, 4> sourceIndices;
    if (failed(resolveSourceIndices(storeOp.getLoc(), rewriter, subViewOp,
                                    storeOp.getIndices(), sourceIndices)))
      return failure();

    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
        storeOp, storeOp.getValue(), subViewOp.getSource(), sourceIndices);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct FoldSubViewPass
    : public PassWrapper<FoldSubViewPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldSubViewPass)

  StringRef getArgument() const final { return "fold-subview"; }

  StringRef getDescription() const final {
    return "Fold memref.subview into affine load/store operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    // Step 1: Folds subviews into their load/store users.
    RewritePatternSet patterns(&getContext());
    patterns.add<LoadOpOfSubViewFolder, StoreOpOfSubViewFolder>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }

    // Step 2: Cleans up dead subview operations that have no remaining users.
    SmallVector<memref::SubViewOp> dead_sub_views;
    getOperation().walk([&](memref::SubViewOp sub_view_op) {
      if (sub_view_op->use_empty()) {
        dead_sub_views.push_back(sub_view_op);
      }
    });

    for (auto sub_view_op : dead_sub_views) {
      sub_view_op.erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createFoldSubViewPass() {
  return std::make_unique<FoldSubViewPass>();
}
