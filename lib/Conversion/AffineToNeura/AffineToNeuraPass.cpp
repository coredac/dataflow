#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::neura;
using namespace mlir::func;

#define GEN_PASS_DEF_LOWERAFFINETONEURA
#include "Conversion/ConversionPasses.h.inc"

namespace {
LogicalResult convertAffineMapToIndices(AffineMap map, ValueRange map_operands,
                                        Location loc, PatternRewriter &rewriter,
                                        SmallVector<Value> &new_indices) {
  new_indices.clear();
  new_indices.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    if (AffineConstantExpr const_expr = dyn_cast<AffineConstantExpr>(expr)) {
      IndexType index_type = rewriter.getIndexType();
      IntegerAttr value_attr =
          rewriter.getIntegerAttr(index_type, const_expr.getValue());
      new_indices.push_back(rewriter.create<neura::ConstantOp>(
          loc, index_type, value_attr));
    } else if (AffineDimExpr dim_expr = dyn_cast<AffineDimExpr>(expr)) {
      if (dim_expr.getPosition() >= map.getNumDims() ||
          dim_expr.getPosition() >=
              map_operands
                  .size()) { // Check against mapOperands size for safety
        return failure();
      }
      new_indices.push_back(map_operands[dim_expr.getPosition()]);
    } else if (AffineSymbolExpr sym_expr = dyn_cast<AffineSymbolExpr>(expr)) {
      unsigned symbol_operand_index = map.getNumDims() + sym_expr.getPosition();
      if (symbol_operand_index >= map_operands.size()) {
        return failure();
      }
      new_indices.push_back(map_operands[symbol_operand_index]);
    } else {
      // For more complex affine expressions (e.g., d0 + c1),
      // materialize the result using affine.apply.
      // This is a temporary workaround for complex expressions.
      // TODO: Handle more complex expressions.
      llvm::errs() << "[affine2neura] Complex affine expression: " << expr
                   << "\n";
      AffineMap single_result_map = AffineMap::get(
          map.getNumDims(), map.getNumSymbols(), expr, rewriter.getContext());
      Value complexIndex = rewriter.create<affine::AffineApplyOp>(
          loc, single_result_map, map_operands);
      new_indices.push_back(complexIndex);
    }
  }
  return success();
}

struct AffineLoadLowering : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineLoadOp load_op,
                                PatternRewriter &rewriter) const override {
    Location loc = load_op.getLoc();
    auto memref = load_op.getMemref();
    AffineMap map = load_op.getAffineMap();
    ValueRange map_operands = load_op.getMapOperands();
    // Gets the indices for the load operation
    SmallVector<Value> new_indices;
    if (failed(convertAffineMapToIndices(map, map_operands, loc, rewriter,
                                         new_indices))) {
      return load_op.emitError(
          "[affine2neura] Failed to convert affine map to indices");
    }

    MemRefType memref_type = dyn_cast<MemRefType>(memref.getType());
    if (!memref_type) {
      return load_op.emitError(
          "[affine2neura] Base of load is not a MemRefType");
    }
    if (new_indices.size() != static_cast<size_t>(memref_type.getRank())) {
      return load_op.emitError(
                 "[affine2neura] Number of indices from affine map (")
             << new_indices.size() << ") does not match memref rank ("
             << memref_type.getRank() << ")";
    }

    // Create the neura.load_indexed operation
   LoadIndexedOp new_load_op = rewriter.create<neura::LoadIndexedOp>(
        loc, load_op.getType(), memref, ValueRange{new_indices});

    rewriter.replaceOp(load_op, new_load_op.getResult());
    return success();
  }
};

struct AffineStoreLowering : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineStoreOp store_op,
                                PatternRewriter &rewriter) const override {
    Location loc = store_op.getLoc();
    auto memref = store_op.getMemref();
    Value value = store_op.getValueToStore();
    AffineMap map = store_op.getAffineMap();
    ValueRange mapOperands = store_op.getMapOperands();

    SmallVector<Value> newIndices;
    if (failed(convertAffineMapToIndices(map, mapOperands, loc, rewriter,
                                         newIndices))) {
      return store_op.emitError(
          "[affine2neura] Failed to convert affine map to indices");
    }

    MemRefType memRefType = dyn_cast<MemRefType>(memref.getType());
    if (!memRefType) {
      return store_op.emitError(
          "[affine2neura] Base of store is not a MemRefType");
    }
    if (newIndices.size() != static_cast<size_t>(memRefType.getRank())) {
      return store_op.emitError(
                 "[affine2neura] Number of indices from affine map (")
             << newIndices.size() << ") does not match memref rank ("
             << memRefType.getRank() << ")";
    }

    rewriter.create<neura::StoreIndexedOp>(loc, value, memref,
                                           ValueRange{newIndices});
    rewriter.eraseOp(store_op);
    return success();
  }
};

struct AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineApplyOp apply_op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = apply_op.getAffineMap();
    ValueRange operands = apply_op.getMapOperands();
    Location loc = apply_op.getLoc();

    // AffineMap can have multiple results when used in affine.for or affine.if,
    // but AffineApplyOp always has exactly one result.
    // Example with multiple results (in affine.for context):
    //   affine_map<(d0, d1) -> (d0 + 1, d1 * 2)>
    // However, AffineApplyOp would use single-result maps like:
    //   affine_map<(d0) -> (d0 + 1)>
    if (map.getNumResults() != 1) {
      return apply_op.emitError(
          "[affine2neura] AffineApplyOp must have a single result");
    }

    AffineExpr expr = map.getResult(0);
    // Handle simple affine expressions like d0 + cst
    // TODO: Handle more complex expressions
    if (isa<AffineBinaryOpExpr>(expr)) {
      AffineBinaryOpExpr bin_expr = dyn_cast<AffineBinaryOpExpr>(expr);
      if (bin_expr.getKind() == AffineExprKind::Add) {
        if (isa<AffineDimExpr>(bin_expr.getLHS())) {
          AffineDimExpr dim = dyn_cast<AffineDimExpr>(bin_expr.getLHS());
          if (isa<AffineConstantExpr>(bin_expr.getRHS())) {
            AffineConstantExpr cst =
                dyn_cast<AffineConstantExpr>(bin_expr.getRHS());
            neura::ConstantOp cstVal = rewriter.create<neura::ConstantOp>(
                loc, rewriter.getIndexType(),
                rewriter.getIntegerAttr(rewriter.getIndexType(),
                                        cst.getValue()));
            neura::AddOp addOp = rewriter.create<neura::AddOp>(
                loc, cstVal.getType(), operands[dim.getPosition()], cstVal);
            rewriter.replaceOp(apply_op, addOp.getResult());
            return success();
          }
        }
      }
    }

    // You can add more cases here for different affine expressions
    // For now, we will just emit an error for unsupported expressions.
    return apply_op.emitError("[affine2neura] Unsupported complex affine "
                              "expression in AffineApplyOp.\n")
           << "Only simple affine expressions like d0 + cst are supported.\n";
  }
};

struct AffineForLowering : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    Location loc = for_op.getLoc();

    // Extract loop bounds - must be constant for now
    if (!for_op.hasConstantLowerBound() || !for_op.hasConstantUpperBound()) {
      return for_op.emitError(
          "[affine2neura] Non-constant loop bounds not supported yet");
    }

    int64_t lower_bound = for_op.getConstantLowerBound();
    int64_t upper_bound = for_op.getConstantUpperBound();
    int64_t step = for_op.getStepAsInt();

    // For now, always create a grant_once for each loop
    // TODO: optimize nested loops to reuse parent's valid signal
    Type i1_type = rewriter.getI1Type();
    Value parent_valid = rewriter.create<neura::GrantOnceOp>(
        loc, i1_type, /*value=*/Value(), /*constant_value=*/nullptr);

    // Create loop_control operation
    auto index_type = rewriter.getIndexType();
    
    auto loop_control = rewriter.create<neura::LoopControlOp>(
        loc,
        /*resultTypes=*/TypeRange{index_type, i1_type},
        /*parentValid=*/parent_valid,
        /*iterationType=*/rewriter.getStringAttr("increment"),
        /*start=*/rewriter.getI64IntegerAttr(lower_bound),
        /*end=*/rewriter.getI64IntegerAttr(upper_bound),
        /*step=*/rewriter.getI64IntegerAttr(step));

    Value loop_index = loop_control.getResult(0);
    // Value loop_valid = loop_control.getResult(1);  // Will be used for nested loops

    // Replace uses of the induction variable
    for_op.getInductionVar().replaceAllUsesWith(loop_index);

    // Inline the body operations before the for_op
    Block &body_block = for_op.getRegion().front();
    Operation *terminator = body_block.getTerminator();
    rewriter.eraseOp(terminator);  // Remove affine.yield first
    
    rewriter.inlineBlockBefore(&body_block, for_op.getOperation(),
                               body_block.getArguments());
    
    // Erase the for_op
    rewriter.eraseOp(for_op);

    return success();
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
    ModuleOp module_op = getOperation();
    MLIRContext *context = module_op.getContext();

    module_op.walk([&](func::FuncOp func_op) {
      // Check if function targets neura accelerator, or apply to all if no attribute
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target = func_op->getAttrOfType<StringAttr>(
            mlir::accel::kAcceleratorAttr);
        if (!target || target.getValue() != mlir::accel::kNeuraTarget) {
          return;  // Skip this function
        }
      }
      // If no accelerator attribute, apply the pass anyway (for testing)
      
      RewritePatternSet patterns(context);
      patterns.add<AffineForLowering, AffineLoadLowering, 
                   AffineStoreLowering, AffineApplyLowering>(context);

      if (failed(applyPatternsGreedily(func_op.getOperation(),
                                       std::move(patterns)))) {
        func_op.emitError("[affine2neura] Failed to lower affine "
                          "operations to Neura dialect");
        signalPassFailure();
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerAffineToNeuraPass() {
  return std::make_unique<LowerAffineToNeuraPass>();
}