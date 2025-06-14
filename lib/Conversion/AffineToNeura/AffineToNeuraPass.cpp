#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::func;

#define GEN_PASS_DEF_LOWERAFFINETONEURA
#include "Conversion/ConversionPasses.h.inc"

namespace {
struct AffineLoadLowering : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto memref = loadOp.getMemref();
    AffineMap map = loadOp.getAffineMap();
    ValueRange mapOperands = loadOp.getMapOperands();
    // Get the indices for the load operation
    SmallVector<Value, 4> newIndices;
    newIndices.reserve(map.getNumResults());
    llvm::errs() << "Lowering affine load operation: " << loadOp << "\n";
    llvm::errs() << "Number of results in affine map: " << map.getNumResults()
                 << "\n";
    for (auto expr : map.getResults()) {
      llvm::errs() << "Map expr: " << expr << "\n";
    }

    for (AffineExpr expr : map.getResults()) {
      if (expr.isa<AffineConstantExpr>()) {
        auto constExpr = expr.cast<AffineConstantExpr>();
        auto indexType = rewriter.getIndexType();
        auto valueAttr =
            rewriter.getIntegerAttr(indexType, constExpr.getValue());
        newIndices.push_back(rewriter.create<neura::ConstantOp>(
            loc, indexType, valueAttr, nullptr));
      } else if (expr.isa<AffineDimExpr>()) {
        auto dimExpr = expr.cast<AffineDimExpr>();
        if (dimExpr.getPosition() >= map.getNumDims() ||
            dimExpr.getPosition() >=
                mapOperands
                    .size()) { // Check against mapOperands size for safety
          return loadOp.emitError(
              "affine map dimension out of bounds for map operands");
        }
        newIndices.push_back(mapOperands[dimExpr.getPosition()]);
      } else if (expr.isa<AffineSymbolExpr>()) {
        auto symExpr = expr.cast<AffineSymbolExpr>();
        unsigned symbolOperandIndex = map.getNumDims() + symExpr.getPosition();
        if (symbolOperandIndex >= mapOperands.size()) {
          return loadOp.emitError(
              "affine map symbol out of bounds for map operands");
        }
        newIndices.push_back(mapOperands[symbolOperandIndex]);
      } else {
        // For more complex affine expressions (e.g., d0 + c1),
        // materialize the result using affine.apply.
        // neura.load_indexed expects individual index values.
        // This is a temporary workaround for complex expressions.
        llvm::errs() << "Complex affine expression: " << expr << "\n";
        AffineMap singleResultMap = AffineMap::get(
            map.getNumDims(), map.getNumSymbols(), expr, rewriter.getContext());
        Value complexIndex = rewriter.create<affine::AffineApplyOp>(
            loc, singleResultMap, mapOperands);
        newIndices.push_back(complexIndex);
      }
    }

    auto memRefType = memref.getType().cast<MemRefType>();
    if (!memRefType) {
      return loadOp.emitError("base of load is not a MemRefType");
    }
    if (newIndices.size() != static_cast<size_t>(memRefType.getRank())) {
      return loadOp.emitError("number of indices from affine map (")
             << newIndices.size() << ") does not match memref rank ("
             << memRefType.getRank() << ")";
    }

    // Create the neura.load_indexed operation
    auto newLoadOp = rewriter.create<neura::LoadIndexedOp>(
        loc, loadOp.getType(), memref, ValueRange{newIndices}, nullptr);

    rewriter.replaceOp(loadOp, newLoadOp.getResult());
    return success();
  }
};

struct AffineStoreLowering : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto memref = storeOp.getMemref();
    auto value = storeOp.getValueToStore();
    AffineMap map = storeOp.getAffineMap();
    ValueRange mapOperands = storeOp.getMapOperands();

    SmallVector<Value, 4> newIndices;
    newIndices.reserve(map.getNumResults());

    for (AffineExpr expr : map.getResults()) {
      if (expr.isa<AffineConstantExpr>()) {
        auto constExpr = expr.cast<AffineConstantExpr>();
        auto indexType = rewriter.getIndexType();
        auto valueAttr =
            rewriter.getIntegerAttr(indexType, constExpr.getValue());
        newIndices.push_back(rewriter.create<neura::ConstantOp>(
            loc, indexType, valueAttr, nullptr));
      } else if (expr.isa<AffineDimExpr>()) {
        auto dimExpr = expr.cast<AffineDimExpr>();
        if (dimExpr.getPosition() >= map.getNumDims() ||
            dimExpr.getPosition() >= mapOperands.size()) {
          return storeOp.emitError(
              "affine map dimension out of bounds for map operands");
        }
        newIndices.push_back(mapOperands[dimExpr.getPosition()]);
      } else if (expr.isa<AffineSymbolExpr>()) {
        auto symExpr = expr.cast<AffineSymbolExpr>();
        unsigned symbolOperandIndex = map.getNumDims() + symExpr.getPosition();
        if (symbolOperandIndex >= mapOperands.size()) {
          return storeOp.emitError(
              "affine map symbol out of bounds for map operands");
        }
        newIndices.push_back(mapOperands[symbolOperandIndex]);
      } else {
        // For more complex affine expressions, materialize the result using
        // affine.apply. This is a temporary workaround for complex expressions.
        AffineMap singleResultMap = AffineMap::get(
            map.getNumDims(), map.getNumSymbols(), expr, rewriter.getContext());
        Value complexIndex = rewriter.create<affine::AffineApplyOp>(
            loc, singleResultMap, mapOperands);
        newIndices.push_back(complexIndex);
      }
    }

    auto memRefType = memref.getType().cast<MemRefType>();
    if (!memRefType) {
      return storeOp.emitError("base of store is not a MemRefType");
    }
    if (newIndices.size() != static_cast<size_t>(memRefType.getRank())) {
      return storeOp.emitError("number of indices from affine map (")
             << newIndices.size() << ") does not match memref rank ("
             << memRefType.getRank() << ")";
    }

    rewriter.create<neura::StoreIndexedOp>(loc, value, memref,
                                           ValueRange{newIndices}, nullptr);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct AffineForLowering : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto loc = forOp.getLoc();
    auto indexType = rewriter.getIndexType();

    // 1. Extract loop parameters (lower bound, upper bound, step)
    Value lowerBoundVal;
    if (forOp.hasConstantLowerBound()) {
      int lowerBoundConstant = forOp.getConstantLowerBound();
      auto lowerBoundAttr =
          rewriter.getIntegerAttr(indexType, lowerBoundConstant);
      lowerBoundVal = rewriter.create<neura::ConstantOp>(
          loc, indexType, lowerBoundAttr, nullptr);
    } else {
      // If the lower bound is not constant, we need to use affine.apply
      // This is a temporary workaround for non-constant lower bounds.
      llvm::errs() << "Using affine.apply for unconstant lower bound\n";
      affine::AffineBound lowerBound = forOp.getLowerBound();
      AffineMap lowerBoundMap = lowerBound.getMap();
      ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
      lowerBoundVal = rewriter.create<affine::AffineApplyOp>(
          loc, lowerBoundMap, lowerBoundOperands);
    }

    Value upperBoundVal;
    if (forOp.hasConstantUpperBound()) {
      int upperBoundConstant = forOp.getConstantUpperBound();
      auto upperBoundAttr =
          rewriter.getIntegerAttr(indexType, upperBoundConstant);
      upperBoundVal = rewriter.create<neura::ConstantOp>(
          loc, indexType, upperBoundAttr, nullptr);
    } else {
      // For non-constant upper bounds, we also use affine.apply
      llvm::errs() << "Using affine.apply for unconstant upper bound\n";
      affine::AffineBound upperBound = forOp.getUpperBound();
      AffineMap upperBoundMap = upperBound.getMap();
      ValueRange upperBoundOperands = forOp.getUpperBoundOperands();
      upperBoundVal = rewriter.create<affine::AffineApplyOp>(
          loc, upperBoundMap, upperBoundOperands);
    }

    auto stepAttr = rewriter.getIntegerAttr(indexType, forOp.getStep());
    Value stepVal =
        rewriter.create<neura::ConstantOp>(loc, indexType, stepAttr, nullptr);
    llvm::errs() << "lower bound: " << lowerBoundVal
                 << ", upper bound: " << upperBoundVal << ", step: " << stepVal
                 << "\n";

    // 2. Block structure
    Block *originBlock = rewriter.getInsertionBlock();
    auto originPoint = rewriter.getInsertionPoint();
    Region *parentRegion = originBlock->getParent();

    Block *headerBlock = rewriter.createBlock(
        parentRegion, std::next(Region::iterator(originBlock)), {indexType},
        {loc});
    Block *bodyBlock = rewriter.createBlock(
        parentRegion, std::next(Region::iterator(headerBlock)), {indexType},
        {loc});
    Block *exitBlock = rewriter.createBlock(
        parentRegion, std::next(Region::iterator(bodyBlock)));
    Block *continueBlock = rewriter.splitBlock(originBlock, originPoint);

    // 3. origin -> header
    rewriter.setInsertionPointToEnd(originBlock);
    rewriter.create<neura::Br>(loc, ValueRange{lowerBoundVal}, headerBlock);

    // 4. header: loop_control
    rewriter.setInsertionPointToEnd(headerBlock);
    SmallVector<Value, 4> bodyArgs;
    bodyArgs.push_back(headerBlock->getArgument(0)); // current index
    // You can add more arguments if needed

    rewriter.create<neura::LoopControlOp>(
        loc,
        headerBlock->getArgument(0), // current index
        stepVal, upperBoundVal, rewriter.getStringAttr("lt"),
        bodyArgs, // passthrough
        bodyBlock, exitBlock);

    // 5. body: clone forOp body, mapping index
    rewriter.setInsertionPointToStart(bodyBlock);
    Value currentIndex = bodyBlock->getArgument(0);
    if (!forOp.getRegion().empty()) {
      Block &sourceBlock = forOp.getRegion().front();
      IRMapping mapping;
      mapping.map(sourceBlock.getArgument(0), currentIndex);
      for (auto &op : llvm::make_range(sourceBlock.begin(),
                                       std::prev(sourceBlock.end()))) {
        Operation *clonedOp = rewriter.clone(op, mapping);
        for (unsigned i = 0; i < op.getNumResults(); ++i)
          mapping.map(op.getResult(i), clonedOp->getResult(i));
      }
    }

    // 6. body 结尾跳 header，传当前 index
    rewriter.setInsertionPointToEnd(bodyBlock);
    rewriter.create<neura::Br>(loc, ValueRange{currentIndex}, headerBlock);

    // 7. exit 跳 continue
    rewriter.setInsertionPointToEnd(exitBlock);
    rewriter.create<neura::Br>(loc, ValueRange{}, continueBlock);

    // 8. 移除原 affine.for
    rewriter.eraseOp(forOp);

    return success();
  }
};

struct AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    AffineMap map = applyOp.getAffineMap();
    ValueRange operands = applyOp.getMapOperands();
    auto loc = applyOp.getLoc();

    if (map.getNumResults() != 1) {
      return applyOp.emitError("AffineApplyOp must have a single result");
    }

    AffineExpr expr = map.getResult(0);
    // d0 + cst
    if (expr.isa<AffineBinaryOpExpr>()) {
      auto binExpr = expr.cast<AffineBinaryOpExpr>();
      if (binExpr.getKind() == AffineExprKind::Add) {
        if (binExpr.getLHS().isa<AffineDimExpr>()) {
          auto dim = binExpr.getLHS().cast<AffineDimExpr>();
          if (binExpr.getRHS().isa<AffineConstantExpr>()) {
            auto cst = binExpr.getRHS().cast<AffineConstantExpr>();
            auto cstVal = rewriter.create<neura::ConstantOp>(
                loc, rewriter.getIndexType(),
                rewriter.getIntegerAttr(rewriter.getIndexType(),
                                        cst.getValue()),
                nullptr);
            auto addOp = rewriter.create<neura::AddOp>(
                loc, cstVal.getType(), operands[dim.getPosition()], cstVal,
                nullptr);
            rewriter.replaceOp(applyOp, addOp.getResult());
            return success();
          }
        }
      }
    }

    // You can add more cases here for different affine expressions
    // For now, we will just emit an error for unsupported expressions.
    return applyOp.emitError(
               "Unsupported complex affine expression in AffineApplyOp.\n")
           << "Only simple affine expressions like d0 + cst are supported.\n";
  }
};

struct LowerAffineToNeuraPass
    : public PassWrapper<LowerAffineToNeuraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAffineToNeuraPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect>();
  }

  StringRef getArgument() const override { return "lower-affine-to-neura"; }
  StringRef getDescription() const override {
    return "Lower affine operations to Neura dialect operations";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp.getContext();

    RewritePatternSet patterns(context);
    patterns.add<AffineLoadLowering, AffineStoreLowering, AffineForLowering,
                 AffineApplyLowering>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      moduleOp.emitError("Failed to lower affine operations to Neura dialect");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerAffineToNeuraPass() {
  return std::make_unique<LowerAffineToNeuraPass>();
}