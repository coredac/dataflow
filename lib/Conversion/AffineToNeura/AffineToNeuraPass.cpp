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
#include "mlir/IR/IRMapping.h"
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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/LogicalResult.h"
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
          loc, index_type, value_attr, nullptr)); // nullptr is for predicated bit
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
        loc, load_op.getType(), memref, ValueRange{new_indices}, nullptr); // nullptr is for predicated bit

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
                                           ValueRange{newIndices}, nullptr); // nullptr is for predicated bit
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
                                        cst.getValue()),
                nullptr); // nullptr is for predicated bit
            neura::AddOp addOp = rewriter.create<neura::AddOp>(
                loc, cstVal.getType(), operands[dim.getPosition()], cstVal,
                nullptr); // nullptr is for predicated bit
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

LogicalResult lowerAffineFor(affine::AffineForOp for_op, OpBuilder &builder,
                             IRMapping &value_mapping) {
  llvm::errs() << "[affine2neura] Lowering AffineForOp: " << for_op << "\n";
  Location loc = for_op.getLoc();
  IndexType index_type = builder.getIndexType();

  // 1 Extract1 loop parameters (lower bound, upper bound, step)
  Value lower_bound_val;
  if (for_op.hasConstantLowerBound()) {
    int64_t lower_bound_constant = for_op.getConstantLowerBound();
    lower_bound_val = builder.create<neura::ConstantOp>(
        loc, index_type, builder.getIndexAttr(lower_bound_constant), nullptr); // nullptr is for predicated bit
  } else {
    // If the lower bound is not constant, we need to use affine.apply
    affine::AffineBound lower_bound = for_op.getLowerBound();
    AffineMap lower_bound_map = lower_bound.getMap();
    ValueRange lower_bound_operands = for_op.getLowerBoundOperands();
    lower_bound_val = builder.create<affine::AffineApplyOp>(
        loc, lower_bound_map, lower_bound_operands);
  }

  Value upper_bound_val;
  if (for_op.hasConstantUpperBound()) {
    int64_t upper_bound_constant = for_op.getConstantUpperBound();
    upper_bound_val = builder.create<neura::ConstantOp>(
        loc, index_type, builder.getIndexAttr(upper_bound_constant), nullptr); // nullptr is for predicated bit
  } else {
    // For non-constant upper bounds, we also use affine.apply
    affine::AffineBound upper_bound = for_op.getUpperBound();
    AffineMap upper_bound_map = upper_bound.getMap();
    ValueRange upper_bound_operands = for_op.getUpperBoundOperands();
    upper_bound_val = builder.create<affine::AffineApplyOp>(
        loc, upper_bound_map, upper_bound_operands);
  }

  Value step_val = builder.create<neura::ConstantOp>(
      loc, index_type, builder.getIndexAttr(for_op.getStepAsInt()), nullptr); // nullptr is for predicated bit

  // 2 Creates the block structure
  Block *origin_block = builder.getInsertionBlock();
  auto origin_point = builder.getInsertionPoint();
  Region *parent_region = origin_block->getParent();

  // 2.1 Creates the header block
  Block *header_block = builder.createBlock(
      parent_region, std::next(Region::iterator(origin_block)), {index_type},
      {loc});
  // 2.2 Creates the body block
  Block *body_block = builder.createBlock(
      parent_region, std::next(Region::iterator(header_block)), {index_type},
      {loc});
  // 2.3 Creates the exit block
  Block *exit_block = builder.createBlock(
      parent_region, std::next(Region::iterator(body_block)));
  // 2.4 Creates the continue block
  Block *continue_block = origin_block->splitBlock(origin_point);

  // 3 Connects the blocks
  // 3.1 Connects origin_block -> header_block
  builder.setInsertionPointToEnd(origin_block);
  builder.create<neura::Br>(loc, ValueRange{lower_bound_val}, header_block);

  // 3.2 Connects header_block -> body_block
  builder.setInsertionPointToEnd(header_block);
  SmallVector<Value> body_args;
  body_args.push_back(header_block->getArgument(0)); // current index
  builder.create<neura::LoopControlOp>(
      loc, header_block->getArgument(0), step_val, upper_bound_val,
      builder.getStringAttr("lt"), body_args, body_block, exit_block);

  // 3.3 Clones the body of the original affine.for operation
  // Assumes the body of the affine.for operation is a single block
  // So we need to guarantee the sequence of handling the nested affine.for
  // operations is correct. (From outermost to innermost)
  builder.setInsertionPointToStart(body_block);
  Value current_index = body_block->getArgument(0);
  if (!for_op.getRegion().empty()) {
    Block &source_block = for_op.getRegion().front();
    IRMapping mapping;
    mapping.map(source_block.getArgument(0), current_index);
    for (Operation &op : llvm::make_range(source_block.begin(),
                                          std::prev(source_block.end()))) {
      Operation *cloned_op = builder.clone(op, mapping);
      for (unsigned i = 0; i < op.getNumResults(); ++i)
        mapping.map(op.getResult(i), cloned_op->getResult(i));
    }
  }

  // 3.4 Connects body_block -> header_block
  builder.setInsertionPointToEnd(body_block);
  builder.create<neura::Br>(loc, ValueRange{current_index}, header_block);

  // 3.5 Connects exit_block -> continue_block
  builder.setInsertionPointToEnd(exit_block);
  builder.create<neura::Br>(loc, ValueRange{}, continue_block);

  builder.setInsertionPointToStart(continue_block);

  for_op.erase();

  return success();
}

affine::AffineForOp findOuterMostAffineFor(func::FuncOp &func_op) {
  // Find the outermost affine.for operation
  affine::AffineForOp top_for_op = nullptr;
  func_op.walk([&](affine::AffineForOp for_op) {
    // Checks if this for_op has any AffineForOp parent
    Operation *parent_op = for_op->getParentOp();
    bool has_affine_for_parent = false;

    while (parent_op) {
      if (isa<affine::AffineForOp>(parent_op)) {
        has_affine_for_parent = true;
        break;
      }
      parent_op = parent_op->getParentOp();
    }

    // If it has no AffineForOp parent, it's a Ftop-level loop
    if (!has_affine_for_parent) {
      top_for_op = for_op;            // Store the found operation
      return WalkResult::interrupt(); // Stop walking
    }

    return WalkResult::advance(); // Continue walking
  });

  return top_for_op; // Return the found operation
}

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
    IRMapping mapping;
    module_op.walk(
        [&](func::FuncOp func_op) {
          if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
            auto target = func_op->getAttrOfType<StringAttr>(
                mlir::accel::kAcceleratorAttr);
            if (target && target.getValue() == mlir::accel::kNeuraTarget) {
              while (affine::AffineForOp outer_for_op =
                         findOuterMostAffineFor(func_op)) {
                llvm::errs()
                    << "[affine2neura] Find outermost affine.for operation: "
                    << outer_for_op << "\n";
                OpBuilder builder(outer_for_op);
                if (failed(lowerAffineFor(outer_for_op, builder, mapping))) {
                  outer_for_op.emitError("[affine2neura] Failed to lower "
                                         "outermost affine.for operation");
                  signalPassFailure();
                }
              }

              RewritePatternSet patterns(context);
              patterns.add<AffineLoadLowering, AffineStoreLowering>(context);

              if (failed(applyPatternsGreedily(func_op.getOperation(),
                                               std::move(patterns)))) {
                func_op.emitError("[affine2neura] Failed to lower affine "
                                    "operations to Neura dialect");
                signalPassFailure();
              }
            }
          }
        });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerAffineToNeuraPass() {
  return std::make_unique<LowerAffineToNeuraPass>();
}