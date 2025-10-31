/*
 * AffineToNeuraPass - Annotated Version for Study
 * 
 * This file provides a detailed annotated version of the AffineToNeura pass
 * implementation. It converts Affine dialect operations (loops, load/store)
 * into Neura dialect operations for CGRA (Coarse-Grained Reconfigurable 
 * Architecture) execution.
 *
 * Key Concepts:
 * =============
 * 
 * 1. Dataflow Semantics:
 *    - Neura dialect uses dataflow execution model
 *    - Operations fire when inputs are available
 *    - Loop control uses valid signals rather than imperative control flow
 *
 * 2. Loop Control Model:
 *    - affine.for (imperative) → neura.loop_control (dataflow)
 *    - Loop bounds stored as attributes (constant at compile time)
 *    - Valid signals control iteration
 *
 * 3. Pattern Rewriting:
 *    - Uses greedy pattern rewriter (bottom-up application)
 *    - Inner loops converted before outer loops
 *    - Each pattern is independent and composable
 */

#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Memref/IR/MemRef.h"
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

/*
 * convertAffineMapToIndices
 * =========================
 * 
 * Converts an AffineMap to a list of index Values suitable for
 * neura.load_indexed/store_indexed operations.
 *
 * AffineMap Structure:
 * -------------------
 * An AffineMap defines index transformations:
 *   map<(d0, d1)[s0] -> (d0 + s0, d1 * 2, 42)>
 *   - d0, d1: dimension operands (loop induction variables)
 *   - s0: symbol operands (parameters)
 *   - Results: expressions to compute indices
 *
 * Conversion Strategy:
 * -------------------
 * For each result expression in the AffineMap:
 *   1. Constant expr (42) → neura.constant
 *   2. Dimension expr (d0) → use corresponding operand directly
 *   3. Symbol expr (s0) → use corresponding operand
 *   4. Complex expr (d0 + 1) → create affine.apply (handled by AffineApplyLowering)
 *
 * Why affine.apply for complex expressions?
 * ----------------------------------------
 * - Allows progressive lowering: affine.apply can later be converted
 * - Separates concerns: each pattern handles one transformation
 * - Enables fallback path: complex expressions can go through affine→scf→neura
 *
 * Parameters:
 * ----------
 * @param map: The AffineMap defining index transformations
 * @param map_operands: Values for dimensions and symbols (d0, d1, ..., s0, s1, ...)
 * @param loc: Source location for new operations
 * @param rewriter: PatternRewriter for creating operations
 * @param new_indices: [OUT] Computed index values
 *
 * Returns:
 * -------
 * success() if all expressions converted successfully
 * failure() if operand indices out of bounds
 */
LogicalResult convertAffineMapToIndices(AffineMap map, ValueRange map_operands,
                                        Location loc, PatternRewriter &rewriter,
                                        SmallVector<Value> &new_indices) {
  // Clear and reserve space for efficiency
  new_indices.clear();
  new_indices.reserve(map.getNumResults());
  
  // Process each result expression in the AffineMap
  // Example: map<(d0, d1) -> (d0, d1 + 1, 0)> has 3 results
  for (AffineExpr expr : map.getResults()) {
    
    // Case 1: Constant Expression
    // ---------------------------
    // Example: affine_map<() -> (42)>
    // Result: Creates neura.constant with value 42
    if (AffineConstantExpr const_expr = dyn_cast<AffineConstantExpr>(expr)) {
      IndexType index_type = rewriter.getIndexType();
      IntegerAttr value_attr =
          rewriter.getIntegerAttr(index_type, const_expr.getValue());
      new_indices.push_back(rewriter.create<neura::ConstantOp>(
          loc, index_type, value_attr));
    } 
    
    // Case 2: Dimension Expression
    // ---------------------------
    // Example: affine_map<(d0, d1) -> (d0)>  // d0 is dimension 0
    // Result: Uses the first operand directly (e.g., loop index %i)
    else if (AffineDimExpr dim_expr = dyn_cast<AffineDimExpr>(expr)) {
      // Safety check: dimension index must be valid
      if (dim_expr.getPosition() >= map.getNumDims() ||
          dim_expr.getPosition() >=
              map_operands
                  .size()) { // Checks against mapOperands size for safety.
        return failure();
      }
      // Directly use the operand corresponding to this dimension
      new_indices.push_back(map_operands[dim_expr.getPosition()]);
    } 
    
    // Case 3: Symbol Expression
    // -------------------------
    // Example: affine_map<(d0)[s0] -> (s0)>  // s0 is symbol 0
    // Result: Uses the symbol operand (parameters passed to the map)
    // 
    // Symbol operands come after dimension operands in map_operands:
    //   map_operands = [dim0, dim1, ..., dimN, sym0, sym1, ..., symM]
    else if (AffineSymbolExpr sym_expr = dyn_cast<AffineSymbolExpr>(expr)) {
      unsigned symbol_operand_index = map.getNumDims() + sym_expr.getPosition();
      if (symbol_operand_index >= map_operands.size()) {
        return failure();
      }
      new_indices.push_back(map_operands[symbol_operand_index]);
    } 
    
    // Case 4: Complex Expression
    // --------------------------
    // Example: affine_map<(d0) -> (d0 + 1)>, affine_map<(d0, d1) -> (d0 * 2)>
    // Result: Creates affine.apply operation to compute the result
    //
    // Why not expand complex expressions here?
    // ----------------------------------------
    // 1. Separation of concerns: Let AffineApplyLowering handle it
    // 2. Progressive lowering: affine.apply → neura operations step by step
    // 3. Fallback path: If direct lowering fails, can use affine→scf→neura
    else {
      // For more complex affine expressions (e.g., d0 + c1),
      // materializes the result using affine.apply.
      // This is a temporary workaround for complex expressions.
      // TODO: Handle more complex expressions.
      llvm::errs() << "[affine2neura] Complex affine expression: " << expr
                   << "\n";
      
      // Create a single-result AffineMap for this expression
      // The created affine.apply will be converted by AffineApplyLowering
      AffineMap single_result_map = AffineMap::get(
          map.getNumDims(), map.getNumSymbols(), expr, rewriter.getContext());
      Value complexIndex = rewriter.create<affine::AffineApplyOp>(
          loc, single_result_map, map_operands);
      new_indices.push_back(complexIndex);
    }
  }
  return success();
}

/*
 * AffineLoadLowering
 * ==================
 *
 * Pattern to convert affine.load to neura.load_indexed.
 *
 * Transformation:
 * --------------
 * Before:
 *   %v = affine.load %memref[map(%i, %j)] : memref<10x10xf32>
 *
 * After:
 *   %idx0 = <computed from map>
 *   %idx1 = <computed from map>
 *   %v = neura.load_indexed %memref[%idx0, %idx1] : memref<10x10xf32>
 *
 * Key Differences:
 * ---------------
 * - affine.load: Uses AffineMap for index calculation
 * - neura.load_indexed: Uses explicit index Values
 *
 * Why this transformation?
 * -----------------------
 * - Neura dialect doesn't support AffineMap (dataflow semantics)
 * - Explicit indices allow hardware to schedule operations independently
 * - Each index calculation becomes a separate dataflow operation
 */
struct AffineLoadLowering : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern<affine::AffineLoadOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineLoadOp load_op,
                                PatternRewriter &rewriter) const override {
    Location loc = load_op.getLoc();
    auto memref = load_op.getMemref();
    AffineMap map = load_op.getAffineMap();
    ValueRange map_operands = load_op.getMapOperands();
    
    // Step 1: Convert AffineMap to explicit index Values
    // Gets the indices for the load operation.
    SmallVector<Value> new_indices;
    if (failed(convertAffineMapToIndices(map, map_operands, loc, rewriter,
                                         new_indices))) {
      return load_op.emitError(
          "[affine2neura] Failed to convert affine map to indices");
    }

    // Step 2: Validate memref type and indices
    // ----------------------------------------
    MemRefType memref_type = dyn_cast<MemRefType>(memref.getType());
    if (!memref_type) {
      return load_op.emitError(
          "[affine2neura] Base of load is not a MemRefType");
    }
    
    // Number of indices must match memref rank
    // Example: memref<10x20xf32> requires exactly 2 indices
    if (new_indices.size() != static_cast<size_t>(memref_type.getRank())) {
      return load_op.emitError(
                 "[affine2neura] Number of indices from affine map (")
             << new_indices.size() << ") does not match memref rank ("
             << memref_type.getRank() << ")";
    }

    // Step 3: Create neura.load_indexed operation
    // Creates the neura.load_indexed operation.
    // 
    // neura.load_indexed semantics:
    // - Fires when all indices are available (dataflow)
    // - No side effects (pure load)
    // - Result available when memory access completes
   LoadIndexedOp new_load_op = rewriter.create<neura::LoadIndexedOp>(
        loc, load_op.getType(), memref, ValueRange{new_indices});

    // Step 4: Replace original operation
    // All uses of the load result are updated automatically
    rewriter.replaceOp(load_op, new_load_op.getResult());
    return success();
  }
};

/*
 * AffineStoreLowering
 * ===================
 *
 * Pattern to convert affine.store to neura.store_indexed.
 *
 * Transformation:
 * --------------
 * Before:
 *   affine.store %value, %memref[map(%i, %j)] : memref<10x10xf32>
 *
 * After:
 *   %idx0 = <computed from map>
 *   %idx1 = <computed from map>
 *   neura.store_indexed %value to %memref[%idx0, %idx1] : memref<10x10xf32>
 *
 * Similar to AffineLoadLowering but for stores.
 * Key difference: store has no result value.
 */
struct AffineStoreLowering : public OpRewritePattern<affine::AffineStoreOp> {
  using OpRewritePattern<affine::AffineStoreOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineStoreOp store_op,
                                PatternRewriter &rewriter) const override {
    Location loc = store_op.getLoc();
    auto memref = store_op.getMemref();
    Value value = store_op.getValueToStore();
    AffineMap map = store_op.getAffineMap();
    ValueRange mapOperands = store_op.getMapOperands();

    // Convert AffineMap to explicit indices
    SmallVector<Value> newIndices;
    if (failed(convertAffineMapToIndices(map, mapOperands, loc, rewriter,
                                         newIndices))) {
      return store_op.emitError(
          "[affine2neura] Failed to convert affine map to indices");
    }

    // Validate memref and indices
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

    // Create neura.store_indexed (no result)
    rewriter.create<neura::StoreIndexedOp>(loc, value, memref,
                                           ValueRange{newIndices});
    // Erase original store operation
    rewriter.eraseOp(store_op);
    return success();
  }
};

/*
 * AffineApplyLowering
 * ===================
 *
 * Pattern to convert affine.apply to neura operations for simple expressions.
 *
 * Background:
 * ----------
 * affine.apply evaluates an AffineMap and returns the result:
 *   %result = affine.apply affine_map<(d0) -> (d0 + 5)>(%i)
 *
 * This pattern handles simple cases that can be directly lowered to neura ops.
 *
 * Supported Expressions:
 * ---------------------
 * Currently supports: d0 + constant
 * Example: affine_map<(d0) -> (d0 + 5)> → neura.add(%d0, neura.constant(5))
 *
 * Unsupported (will fail):
 * -----------------------
 * - Multiplication: d0 * 2
 * - Division: d0 / 2
 * - Multiple dimensions: d0 + d1
 * - Modulo: d0 mod 16
 *
 * Fallback Strategy:
 * -----------------
 * When unsupported, user should:
 * 1. Use --lower-affine-to-scf first (affine → SCF dialect)
 * 2. Then --lower-scf-to-neura (SCF → Neura dialect)
 * This provides full affine expression support.
 */
struct AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineApplyOp apply_op,
                                PatternRewriter &rewriter) const override {
    AffineMap map = apply_op.getAffineMap();
    ValueRange operands = apply_op.getMapOperands();
    Location loc = apply_op.getLoc();

    // Sanity check: affine.apply always has exactly one result
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
    
    // Pattern matching for supported expressions
    // Handles simple affine expressions like d0 + cst.
    // TODO: Handle more complex expressions.
    
    // Check if expression is a binary operation
    if (isa<AffineBinaryOpExpr>(expr)) {
      AffineBinaryOpExpr bin_expr = dyn_cast<AffineBinaryOpExpr>(expr);
      
      // Case: Addition (d0 + cst)
      // ------------------------
      if (bin_expr.getKind() == AffineExprKind::Add) {
        // Left side should be a dimension (e.g., d0)
        if (isa<AffineDimExpr>(bin_expr.getLHS())) {
          AffineDimExpr dim = dyn_cast<AffineDimExpr>(bin_expr.getLHS());
          
          // Right side should be a constant (e.g., 5)
          if (isa<AffineConstantExpr>(bin_expr.getRHS())) {
            AffineConstantExpr cst =
                dyn_cast<AffineConstantExpr>(bin_expr.getRHS());
            
            // Create neura operations: constant + add
            // Example: d0 + 5 becomes:
            //   %c5 = neura.constant 5 : index
            //   %result = neura.add %d0, %c5 : index
            neura::ConstantOp cstVal = rewriter.create<neura::ConstantOp>(
                loc, rewriter.getIndexType(),
                rewriter.getIntegerAttr(rewriter.getIndexType(),
                                        cst.getValue()));
            neura::AddOp addOp = rewriter.create<neura::AddOp>(
                loc, cstVal.getType(), operands[dim.getPosition()], cstVal);
            
            // Replace affine.apply with add result
            rewriter.replaceOp(apply_op, addOp.getResult());
            return success();
          }
        }
      }
      
      // More cases can be added here:
      // - Subtraction: d0 - cst
      // - Multiplication by power of 2: d0 * 4 (can use shift)
      // - etc.
    }

    // Unsupported expression - fail with helpful message
    // You can add more cases here for different affine expressions.
    // For now, we will just emit an error for unsupported expressions.
    return apply_op.emitError("[affine2neura] Unsupported complex affine "
                              "expression in AffineApplyOp.\n")
           << "Only simple affine expressions like d0 + cst are supported.\n";
  }
};

/*
 * AffineForLowering
 * =================
 *
 * Pattern to convert affine.for loops to neura dataflow operations.
 *
 * Imperative vs Dataflow Loop Models:
 * -----------------------------------
 * 
 * Affine (Imperative):
 *   affine.for %i = 0 to N step 2 {
 *     %v = affine.load %A[%i]
 *     affine.store %v, %B[%i]
 *   }
 * 
 * Control flow: PC-based, sequential execution
 * Loop control: Compare, branch instructions
 * 
 * Neura (Dataflow):
 *   %grant = neura.grant_once            // Start signal
 *   %i, %valid = neura.loop_control(%grant) <{start=0, end=N, step=2}>
 *   %v = neura.load_indexed %A[%i]      // Fires when %i available
 *   neura.store_indexed %v to %B[%i]    // Fires when %v, %i available
 * 
 * Control flow: Token-based, operations fire when inputs ready
 * Loop control: Valid signals propagate through dataflow graph
 *
 * Transformation Strategy:
 * -----------------------
 * 1. Create grant_once: Provides initial valid signal
 * 2. Create loop_control: Generates iteration indices and valid signals
 * 3. Inline loop body: Operations execute dataflow-style
 * 4. Replace induction variable: Use loop_control index output
 *
 * Loop Control Semantics:
 * ----------------------
 * neura.loop_control(%parent_valid) <{start, end, step, type}>
 *   → (%index, %valid)
 *
 * - Inputs:
 *   * parent_valid: Signal indicating when to start/continue
 * - Outputs:
 *   * index: Current iteration value
 *   * valid: Signal indicating iteration is active
 * - Attributes:
 *   * start, end, step: Loop bounds (must be constant)
 *   * type: "increment" or "decrement"
 *
 * Why Attributes for Bounds?
 * -------------------------
 * - Dataflow scheduling: Hardware needs static loop bounds
 * - Compile-time analysis: Enable loop unrolling, pipelining
 * - Resource allocation: Calculate buffer sizes, etc.
 *
 * Design Decision: No Dynamic Bounds Support
 * ------------------------------------------
 * Dynamic loop bounds (determined at runtime) are not supported because:
 * 1. CGRA hardware configuration requires compile-time known loop structure
 * 2. Static bounds enable critical hardware optimizations (pipelining, unrolling)
 * 3. If dynamic loops are needed:
 *    - Execute on host CPU instead of CGRA
 *    - Or use conservative maximum bounds with early exit at runtime
 *
 * Nested Loop Handling:
 * --------------------
 * Current: Each loop gets independent grant_once
 *   Outer: grant_once → loop_control → body
 *   Inner: grant_once → loop_control → body
 *
 * This works but creates redundant control signals.
 *
 * Future optimization:
 *   Outer: grant_once → loop_control → body
 *                          ↓ (reuse valid signal)
 *   Inner:               loop_control → body
 *
 * TODO: Optimize nested loops to reuse parent's valid signal.
 * This requires dataflow analysis to identify parent-child relationships.
 */
struct AffineForLowering : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    Location loc = for_op.getLoc();

    // Step 1: Extract and validate loop bounds
    // ----------------------------------------
    // Extracts loop bounds - must be constant for now.
    // 
    // Why constant bounds only?
    // - Neura loop_control uses attributes (compile-time constants)
    // - Hardware schedulers need static loop bounds
    // - Dynamic bounds would require Value operands (future work)
    if (!for_op.hasConstantLowerBound() || !for_op.hasConstantUpperBound()) {
      return for_op.emitError(
          "[affine2neura] Non-constant loop bounds not supported yet");
    }

    int64_t lower_bound = for_op.getConstantLowerBound();
    int64_t upper_bound = for_op.getConstantUpperBound();
    int64_t step = for_op.getStepAsInt();

    // Step 2: Create parent valid signal
    // ----------------------------------
    // For now, always creates a grant_once for each loop.
    // TODO: Optimize nested loops to reuse parent's valid signal.
    //
    // grant_once semantics:
    // - Fires once at the start
    // - Provides initial valid signal to loop_control
    // - Can be gated by a predicate (not used here yet)
    Type i1_type = rewriter.getI1Type();
    Value parent_valid = rewriter.create<neura::GrantOnceOp>(
        loc, i1_type, /*value=*/Value(), /*constant_value=*/nullptr);

    // Step 3: Create loop_control operation
    // -------------------------------------
    // Creates loop_control operation.
    //
    // This is the heart of dataflow loop execution:
    // - Takes parent_valid as input
    // - Outputs (index, valid) for each iteration
    // - Bounds specified as attributes
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
    // Note: loop_control.getResult(1) returns loop_valid signal.
    // loop_valid can be used to gate operations within the loop body.
    // For nested loops, the inner loop's parent_valid should use the outer
    // loop's loop_valid signal instead of creating a new grant_once.
    // This optimization requires dataflow analysis to identify parent-child
    // loop relationships, which is not yet implemented.
    // For now, each loop creates its own independent grant_once signal.

    // Step 4: Replace induction variable
    // ----------------------------------
    // Replaces uses of the induction variable.
    //
    // Original affine.for:
    //   affine.for %i = 0 to N {
    //     %v = affine.load %A[%i]  // Uses induction variable %i
    //   }
    //
    // After transformation:
    //   %i, %valid = neura.loop_control(...)
    //   %v = neura.load_indexed %A[%i]  // Uses loop_control index output
    //
    // replaceAllUsesWith updates all references automatically
    for_op.getInductionVar().replaceAllUsesWith(loop_index);

    // Step 5: Inline loop body
    // -----------------------
    // Inlines the body operations before the for_op.
    //
    // Original structure:
    //   affine.for %i ... {
    //     ^bb0(%i: index):
    //       <body operations>
    //       affine.yield
    //   }
    //
    // After inlining:
    //   %grant = neura.grant_once
    //   %i, %valid = neura.loop_control(...)
    //   <body operations>  // Inlined here
    //
    // Why inline instead of keeping region?
    // - Neura dialect uses flat structure (no imperative control flow)
    // - Operations execute based on data availability (dataflow)
    // - Regions would imply control flow boundaries
    //
    // Pattern application order ensures correctness:
    // - Greedy rewriter applies patterns bottom-up
    // - Inner loops converted first (their operations already lowered)
    // - Then outer loops converted (inner neura ops already in place)
    Block &body_block = for_op.getRegion().front();
    Operation *terminator = body_block.getTerminator();
    rewriter.eraseOp(terminator);  // Removes affine.yield first.
    
    // inlineBlockBefore: Moves operations from body_block to before for_op
    // This preserves SSA dominance:
    // - loop_control defines %i
    // - %i is used by inlined body operations
    // - Correct dominance: loop_control comes before uses
    rewriter.inlineBlockBefore(&body_block, for_op.getOperation(),
                               body_block.getArguments());
    
    // Step 6: Remove original for operation
    // -------------------------------------
    // Erases the for_op.
    // At this point:
    // - Body operations inlined
    // - Induction variable replaced
    // - Loop structure no longer needed
    rewriter.eraseOp(for_op);

    return success();
  }
};

/*
 * LowerAffineToNeuraPass
 * ======================
 *
 * Main pass implementation that orchestrates all pattern applications.
 *
 * Pass Architecture:
 * -----------------
 * MLIR uses a pipeline of passes to progressively lower IR:
 *   Affine Dialect (high-level loops)
 *     ↓ [this pass]
 *   Neura Dialect (dataflow operations)
 *     ↓ [subsequent passes]
 *   Hardware Configuration (CGRA bitstream)
 *
 * Pattern Application Strategy:
 * ----------------------------
 * Uses greedy pattern rewriter:
 * - Applies patterns repeatedly until no more matches
 * - Bottom-up traversal (children before parents)
 * - Ensures inner loops converted before outer loops
 *
 * Why greedy instead of one-shot?
 * - Patterns interact: load/store inside loops
 * - Order matters: inner → outer for nested loops
 * - Flexibility: can add/remove patterns easily
 *
 * Target Functions:
 * ----------------
 * Only applies to functions targeting Neura accelerator:
 * - Check accelerator attribute
 * - Skip functions targeting other accelerators
 * - Apply to all if no attribute (for testing)
 */
struct LowerAffineToNeuraPass
    : public PassWrapper<LowerAffineToNeuraPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAffineToNeuraPass)

  // Register required dialects
  // All dialects used in this pass must be registered
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect,    // Target dialect
                    arith::ArithDialect,      // For arithmetic ops
                    memref::MemRefDialect,    // For memory operations
                    affine::AffineDialect>();  // Source dialect
  }

  // Pass command-line interface
  StringRef getArgument() const override { return "lower-affine-to-neura"; }
  StringRef getDescription() const override {
    return "Lower affine operations to Neura dialect operations";
  }

  // Main pass logic
  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = module_op.getContext();

    // Walk through all functions in the module
    // Applies transformation function-by-function
    module_op.walk([&](func::FuncOp func_op) {
      // Target selection: which functions to transform
      // Checks if function targets neura accelerator, or applies to all if no attribute.
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target = func_op->getAttrOfType<StringAttr>(
            mlir::accel::kAcceleratorAttr);
        if (!target || target.getValue() != mlir::accel::kNeuraTarget) {
          return;  // Skips this function.
        }
      }
      // If no accelerator attribute, applies the pass anyway (for testing).
      
      // Register all rewrite patterns
      // Order doesn't matter - greedy rewriter handles ordering
      RewritePatternSet patterns(context);
      patterns.add<AffineForLowering,      // Convert loops
                   AffineLoadLowering,      // Convert loads
                   AffineStoreLowering,     // Convert stores
                   AffineApplyLowering>     // Convert index calculations
                  (context);

      // Apply patterns greedily
      // Continues until no patterns match (fixed point)
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

/*
 * Pass Factory Function
 * ====================
 *
 * Creates and returns a unique instance of the pass.
 * Called by MLIR pass manager when building pass pipeline.
 *
 * Usage:
 *   PassManager pm(...);
 *   pm.addPass(mlir::createLowerAffineToNeuraPass());
 *   pm.run(module);
 *
 * Or from command line:
 *   mlir-neura-opt input.mlir --lower-affine-to-neura
 */
std::unique_ptr<mlir::Pass> mlir::createLowerAffineToNeuraPass() {
  return std::make_unique<LowerAffineToNeuraPass>();
}

/*
 * Summary of Key Design Decisions:
 * =================================
 *
 * 1. Dataflow over Control Flow:
 *    - Operations fire when inputs ready
 *    - Valid signals instead of PC
 *    - Enables spatial parallelism on CGRA
 *
 * 2. Attribute-based Loop Bounds:
 *    - Compile-time constants enable optimization
 *    - Hardware schedulers can pre-compute iterations
 *    - Design decision: No dynamic bounds (CGRA hardware limitation)
 *
 * 3. Progressive Lowering:
 *    - affine.apply for complex expressions
 *    - Can fallback to affine→scf→neura
 *    - Each pass handles one level of abstraction
 *
 * 4. Independent grant_once per Loop:
 *    - Simple and correct
 *    - Can be optimized: Reuse parent valid for nested loops (requires dataflow analysis)
 *    - Trade-off: Some redundancy for implementation simplicity
 *
 * 5. Greedy Pattern Application:
 *    - Bottom-up ensures inner before outer
 *    - Fixed-point iteration until stable
 *    - Flexible: easy to add new patterns
 *
 * Future Work:
 * ===========
 * - More affine expressions (mul, div, mod, etc.) with direct lowering
 * - Nested loop optimization (reuse parent valid signal, requires dataflow analysis)
 * - Polyhedral analysis for loop transformations
 * - Support for affine.if (conditional execution)
 * 
 * Features Explicitly Not Supported:
 * ==================================
 * - Dynamic loop bounds: Fundamental CGRA hardware limitation, will not be supported
 *   Code requiring dynamic loops should execute on host CPU
 */
