#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
using namespace mlir::neura;

namespace {
// Lowers integer add from mlir.llvm.add to nuera.add. We provide the lowering
// here instead of tablegen due to that mlir.llvm.add uses an EnumProperty
// (IntegerOverflowFlags) defined via MLIR interfaces — which DRR cannot match
// on or extract from.
struct LlvmAddToNeuraAdd : public OpRewritePattern<mlir::LLVM::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, op.getType(), op.getLhs(),
                                              op.getRhs());
    return success();
  }
};

struct LlvmFAddToNeuraFAdd : public OpRewritePattern<mlir::LLVM::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FAddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmFSubToNeuraFSub : public OpRewritePattern<mlir::LLVM::FSubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FSubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op.getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<neura::FSubOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmAndToNeuraAnd : public OpRewritePattern<mlir::LLVM::AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AndOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::AndOp>(op, op.getType(), op.getLhs(),
                                              op.getRhs());
    return success();
  }
};

struct LlvmOrToNeuraOr : public OpRewritePattern<mlir::LLVM::OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::OrOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::OrOp>(op, op.getType(), op.getLhs(),
                                             op.getRhs());
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

struct LlvmSDivToNeuraDiv : public OpRewritePattern<LLVM::SDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SDivOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::DivOp>(op, resultType, lhs, rhs);
    return success();
  }
};

struct LlvmSRemToNeuraRem : public OpRewritePattern<LLVM::SRemOp> {
  using OpRewritePattern<LLVM::SRemOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SRemOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type resultType = op.getType();

    // Create neura.rem operation to replace llvm.srem
    rewriter.replaceOpWithNewOp<neura::RemOp>(op, resultType, lhs, rhs);
    return success();
  }
};

struct LlvmMaxNumToNeuraFMax : public OpRewritePattern<LLVM::MaxNumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MaxNumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type resultType = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(resultType))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMaxOp>(op, resultType, lhs, rhs,
                                               rewriter.getStringAttr("maxnum"));
    return success();
  }
};

struct LlvmMaximumToNeuraFMax : public OpRewritePattern<LLVM::MaximumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MaximumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type resultType = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(resultType))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMaxOp>(op, resultType, lhs, rhs,
                                               rewriter.getStringAttr("maximum"));
    return success();
  }
};

struct LlvmMinNumToNeuraFMin : public OpRewritePattern<LLVM::MinNumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MinNumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type resultType = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(resultType))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMinOp>(op, resultType, lhs, rhs,
                                               rewriter.getStringAttr("minnum"));
    return success();
  }
};

struct LlvmMinimumToNeuraFMin : public OpRewritePattern<LLVM::MinimumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MinimumOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type resultType = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(resultType))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMinOp>(op, resultType, lhs, rhs,
                                               rewriter.getStringAttr("minimum"));
    return success();
  }
};

struct LlvmFDivToNeuraFDiv : public OpRewritePattern<mlir::LLVM::FDivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FDivOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FDivOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmFPToSIToNeuraCast : public OpRewritePattern<mlir::LLVM::FPToSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FPToSIOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getArg();
    Type result_type = op.getType();

    // Creates a cast operation with "fptosi" as the cast type.
    rewriter.replaceOpWithNewOp<neura::CastOp>(op, result_type, input, 
                                               rewriter.getStringAttr("fptosi"));
    return success();
  }
};

struct LlvmSelectToNeuraSel : public OpRewritePattern<LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value cond = op.getCondition();
    Value true_value = op.getTrueValue();
    Value false_value = op.getFalseValue();
    Type result_type = op.getType();

    // neura.sel now follows the same order as llvm.select: (cond, ifTrue, ifFalse)
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, 
                                               cond, true_value, false_value);
    return success();
  }
};

struct LlvmFMulAddToNeuraFMulFAdd : public OpRewritePattern<mlir::LLVM::FMulAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulAddOp op,
                                PatternRewriter &rewriter) const override {
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Value c = op->getOperand(2);
    Type result_type = op->getResult(0).getType();

    // Only matches scalar float.
    if (!mlir::isa<FloatType>(result_type))
      return failure();

    rewriter.replaceOpWithNewOp<neura::FMulFAddOp>(op, result_type, a, b, c);
    return success();
  }
};

// Handles LLVM intrinsic memset operations
struct LlvmMemsetToNeuraOps : public OpRewritePattern<LLVM::MemsetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MemsetOp op,
                                PatternRewriter &rewriter) const override {
    // Get the operands: dest, value, len, isvolatile
    auto dest = op.getDst();
    auto value = op.getVal();
    // Note: len and isvolatile are not used in this simplified implementation
    // but are kept for future enhancement
    (void)op.getLen(); // len - not used in current implementation
    (void)op.getIsVolatile(); // isvolatile - not used in current implementation
    
    // For CGRA, we can implement memset as a simple store operation
    // Create a constant for the value to set (use the actual value if available)
    Attribute valueAttr;
    if (auto constOp = value.getDefiningOp<neura::ConstantOp>()) {
      valueAttr = constOp->getAttr("value");
    } else {
      // Default to 0 if we can't determine the value
      valueAttr = rewriter.getI8IntegerAttr(0);
    }
    
    auto valueConst = rewriter.create<neura::ConstantOp>(
        op.getLoc(), rewriter.getI8Type(), valueAttr);
    
    // Create a store operation to set the memory
    // Note: This is a simplified implementation
    // In a real implementation, we might need to handle the length parameter
    auto storeOp = rewriter.create<neura::StoreOp>(
        op.getLoc(), valueConst, dest);
    
    // Replace the memset with the store operation
    rewriter.replaceOp(op, storeOp->getResults());
    return success();
  }
};

struct LlvmVFMulToNeuraVFMul : public OpRewritePattern<mlir::LLVM::FMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FMulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xf32>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<FloatType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VFMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVMulToNeuraVMul : public OpRewritePattern<mlir::LLVM::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xInt>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<IntegerType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VMulOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVAddToNeuraVAdd : public OpRewritePattern<mlir::LLVM::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::AddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xInt>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<IntegerType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

struct LlvmVFAddToNeuraVFAdd : public OpRewritePattern<mlir::LLVM::FAddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::FAddOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Type result_type = op->getResult(0).getType();

    // Only matches vector<xf32>.
    auto vec_type = mlir::dyn_cast<VectorType>(result_type);
    if (!vec_type || !mlir::isa<FloatType>(vec_type.getElementType()))
      return failure();

    rewriter.replaceOpWithNewOp<neura::VFAddOp>(op, result_type, lhs, rhs);
    return success();
  }
};

// Handles LLVM intrinsic operations like llvm.intr.vector.reduce.add
// These are generic intrinsic calls, not specific op types
struct LlvmVectorReduceAddToNeuraVectorReduceAdd : public RewritePattern {
  LlvmVectorReduceAddToNeuraVectorReduceAdd(MLIRContext *context)
      : RewritePattern("llvm.intr.vector.reduce.add", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Check that we have exactly one operand and one result
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    
    Value input = op->getOperand(0);
    Type result_type = op->getResult(0).getType();

    rewriter.replaceOpWithNewOp<neura::VectorReduceAddOp>(op, result_type, input);
    return success();
  }
};

struct LlvmICmpToNeuraICmp : public OpRewritePattern<LLVM::ICmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ICmpOp op,
                                PatternRewriter &rewriter) const override {
    auto pred = op.getPredicate();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::ICmpOp>(
        op, resultType, lhs, rhs,
        rewriter.getStringAttr(LLVM::stringifyICmpPredicate(pred)));
    return success();
  }
};

struct LlvmFCmpToNeuraFCmp : public OpRewritePattern<LLVM::FCmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FCmpOp op,
                                PatternRewriter &rewriter) const override {
    auto pred = op.getPredicate();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::FCmpOp>(
        op, resultType, lhs, rhs,
        rewriter.getStringAttr(LLVM::stringifyFCmpPredicate(pred)));
    return success();
  }
};

struct LlvmGEPToNeuraGEP : public OpRewritePattern<mlir::LLVM::GEPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::GEPOp op,
                                PatternRewriter &rewriter) const override {
    Value base = op.getBase();
    SmallVector<Value> indexValues;

    for (auto gepIndex : op.getIndices()) {
      if (auto val = gepIndex.dyn_cast<Value>()) {
        indexValues.push_back(val);
      } else if (auto intAttr = gepIndex.dyn_cast<IntegerAttr>()) {
        // Creates constant operation state manually.
        OperationState state(op.getLoc(),
                             neura::ConstantOp::getOperationName());
        state.addAttribute("value", intAttr);
        state.addTypes(rewriter.getIndexType());
        Value cst = rewriter.create(state)->getResult(0);
        indexValues.push_back(cst);
      } else {
        return op.emitOpError("Unsupported GEP index kind");
      }
    }

    rewriter.replaceOpWithNewOp<neura::GEP>(op, op.getType(), base,
                                            indexValues);
    return success();
  }
};

struct LlvmLoadToNeuraLoad : public OpRewritePattern<mlir::LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value ptr = op.getAddr(); // getPointer() is deprecated.
    Type resultType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<neura::LoadOp>(op, resultType, ptr);
    return success();
  }
};

struct LlvmStoreToNeuraStore : public OpRewritePattern<mlir::LLVM::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    Value addr = op.getAddr(); // getPointer() is deprecated
    rewriter.replaceOpWithNewOp<neura::StoreOp>(op, value, addr);
    return success();
  }
};

struct LlvmCondBrToNeuraCondBr : public OpRewritePattern<LLVM::CondBrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CondBrOp op,
                                PatternRewriter &rewriter) const override {
    // Gets the source operation's successors (basic blocks).
    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    // Gets the operands for each destination.
    ValueRange trueOperands = op.getTrueDestOperands();
    ValueRange falseOperands = op.getFalseDestOperands();

    // Creates the new operation with proper successors.
    auto newOp = rewriter.create<neura::CondBr>(
        op.getLoc(),       // Location
        op.getCondition(), // Condition
        trueOperands,      // True destination operands
        falseOperands,     // False destination operands
        trueDest,          // True destination block
        falseDest          // False destination block
    );

    // Replaces the old op with the new one.
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

struct LlvmBrToNeuraBr : public OpRewritePattern<LLVM::BrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::LLVM::BrOp op,
                                PatternRewriter &rewriter) const override {
    // Gets the destination block and its operands.
    Block *dest = op.getDest();
    ValueRange destOperands = op.getDestOperands();

    // Creates the new Neura_Br operation.
    rewriter.replaceOpWithNewOp<neura::Br>(op, destOperands, dest);

    return success();
  }
};

struct LlvmReturnToNeuraReturn : public OpRewritePattern<LLVM::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::ReturnOp>(op, op.getOperands());
    return success();
  }
};

struct LlvmFNegToNeuraFSub : public OpRewritePattern<LLVM::FNegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FNegOp op,
                                PatternRewriter &rewriter) const override {
    // Gets operand.
    Value operand = op.getOperand();
    Type result_type = op.getType();
    Location loc = op.getLoc();

    // Creates constant 0.0.
    auto zero_attr = rewriter.getFloatAttr(result_type, 0.0);
    auto zero_const = rewriter.create<neura::ConstantOp>(loc, result_type, zero_attr);

    // Replaces with 0.0 - operand.
    rewriter.replaceOpWithNewOp<neura::FSubOp>(op, result_type, zero_const, operand);
    return success();
  }
};

struct LlvmSubToNeuraSub : public OpRewritePattern<LLVM::SubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SubOp op,
                                PatternRewriter &rewriter) const override {
    // Gets operands.
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Replaces with neura.sub.
    rewriter.replaceOpWithNewOp<neura::SubOp>(op, result_type, lhs, rhs);
    return success();
  }
};

// TODO: Implement LlvmXOrToNeuraOr - used in ADPCM coder and FFT kernels
//       llvm.xor operations appear in:
//       - adpcm_coder-kernel.mlir (line 104: %87 = llvm.xor %29, %19 : i1)
//       - fft_kernel.mlir (line 19: %11 = llvm.xor %10, %3 : i32)
//       Implementation: xor(a, b) = or(a, b) for boolean values

// TODO: Implement LlvmAndToNeuraMul - used in ADPCM coder and MVT kernels
//       llvm.and operations appear in:
//       - adpcm_coder-kernel.mlir (lines 55, 94: bitwise AND operations)
//       - mvt-kernel.mlir (lines 44, 47, 50, 53: vector and scalar AND operations)
//       Implementation: and(a, b) = mul(a, b) for boolean values

// TODO: Implement LlvmAllocaToNeuraOps - used in DTW kernel
//       llvm.alloca operations appear in:
//       - dtw-kernel-O0.mlir (lines 19-23: multiple stack allocations)
//       Implementation: For CGRA, erase alloca or convert to register allocation

// TODO: Implement LlvmLShrToNeuraShl - used in ADPCM coder/decoder and FFT kernels
//       llvm.lshr operations appear in:
//       - adpcm_coder-kernel.mlir (line 54: %42 = llvm.lshr %40, %7 : i32)
//       - adpcm_decoder-kernel.ll (line 35: %30 = lshr i32 %29, 4)
//       - fft_kernel.mlir (line 67: %49 = llvm.lshr %7, %1 : i32)
//       Implementation: Need proper logical right shift (lshr(x,n) != shl(x,-n))

// TODO: Implement LlvmAShrToNeuraAShr - used in ADPCM coder/decoder kernels
//       llvm.ashr operations appear in:
//       - adpcm_coder-kernel.mlir (lines 57, 63, 70: multiple ashr operations)
//       - adpcm_decoder-kernel.ll (lines 49, 56, 61: ashr i32 %20, 3/1/2)
//       Implementation: Need proper arithmetic right shift (preserves sign bit)


struct LlvmSMaxToNeuraSMax : public OpRewritePattern<LLVM::SMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SMaxOp op,
                                PatternRewriter &rewriter) const override {
    // Gets operands.
    Value lhs = op.getA();
    Value rhs = op.getB();
    Type result_type = op.getType();
    Location loc = op.getLoc();

    // Implements smax(a, b) = a >= b ? a : b
    auto cmp = rewriter.create<neura::ICmpOp>(loc, rewriter.getI1Type(), 
                                             lhs, rhs,
                                             rewriter.getStringAttr("sge"));
    
    // Select: a >= b ? a : b
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, cmp, lhs, rhs);
    return success();
  }
};

// TODO: Implement LlvmAbsToNeuraAbs - used in ADPCM coder kernel
//       llvm.intr.abs operations appear in adpcm_coder-kernel.mlir  
//       Implementation: abs(x) = x >= 0 ? x : -x (using ICmpOp + SelOp)

struct FuncReturnToNeuraReturn : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<neura::ReturnOp>(op, op.getOperands());
    return success();
  }
};

struct LlvmConstantToNeuraConstant : public OpRewritePattern<LLVM::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto attr = op.getValue();

    // Creates operation state manually.
    OperationState state(op.getLoc(), neura::ConstantOp::getOperationName());
    state.addAttribute("value", attr);
    state.addTypes(op.getType());

    // Creates the operation and replaces.
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct LlvmAllocaToNeuraAlloca : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    Value size = op.getArraySize();
    Type resultType = op.getType();

    // Converts the size to neura.data<i32, i1> if it's not already.
    // Assumes the size is already in the right format.
    // Handles type conversion here.

    rewriter.replaceOpWithNewOp<neura::AllocaOp>(op, resultType, size);
    return success();
  }
};

struct LlvmSExtToNeuraSExt : public OpRewritePattern<LLVM::SExtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SExtOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getArg();
    Type resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::SExtOp>(op, resultType, input);
    return success();
  }
};

struct LlvmZExtToNeuraZExt : public OpRewritePattern<LLVM::ZExtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ZExtOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getArg();
    Type resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::ZExtOp>(op, resultType, input);
    return success();
  }
};

struct LlvmTruncToNeuraCast : public OpRewritePattern<LLVM::TruncOp> {
  using OpRewritePattern<LLVM::TruncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::TruncOp op,
                                PatternRewriter &rewriter) const override {
    // Trunc is a simple cast operation
    auto result = rewriter.create<neura::CastOp>(
        op.getLoc(), op.getType(), op.getArg(),
        rewriter.getStringAttr("trunc"));
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct LlvmUDivToNeuraDiv : public OpRewritePattern<LLVM::UDivOp> {
  using OpRewritePattern<LLVM::UDivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::UDivOp op,
                                PatternRewriter &rewriter) const override {
    // UDiv is unsigned division
    auto result = rewriter.create<neura::DivOp>(
        op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct LlvmURemToNeuraRem : public OpRewritePattern<LLVM::URemOp> {
  using OpRewritePattern<LLVM::URemOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::URemOp op,
                                PatternRewriter &rewriter) const override {
    // URem is unsigned remainder
    auto result = rewriter.create<neura::RemOp>(
        op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct LlvmMulToNeuraMul : public OpRewritePattern<LLVM::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::MulOp>(op, resultType, lhs, rhs);
    return success();
  }
};

struct LlvmShlToNeuraShl : public OpRewritePattern<LLVM::ShlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ShlOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type resultType = op.getType();

    rewriter.replaceOpWithNewOp<neura::ShlOp>(op, resultType, lhs, rhs);
    return success();
  }
};

struct LlvmFuncToNeuraFunc : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter &rewriter) const override {

    auto target = op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
    if (!target || target.getValue() != mlir::accel::kNeuraTarget) {
      return failure();
    }

    // Converts LLVMFunctionType to FunctionType.
    auto llvmFuncType = op.getFunctionType();
    auto funcType = rewriter.getFunctionType(llvmFuncType.getParams(),
                                             llvmFuncType.getReturnType());

    // Creates the new func.func operation using OperationState to have full
    // control.
    OperationState state(op.getLoc(), func::FuncOp::getOperationName());
    state.addAttribute("sym_name", rewriter.getStringAttr(op.getName()));
    state.addAttribute("function_type", TypeAttr::get(funcType));

    // Copies ALL attributes from the original llvm.func exactly as they are.
    // Skips function type and name attributes as they are handled separately.
    SmallVector<NamedAttribute> attrs;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "function_type" || attr.getName() == "sym_name") {
        continue;
      }
      attrs.push_back(attr);
    }
    state.addAttributes(attrs);


    // Adds the function body region.
    state.addRegion();


    auto newFunc = cast<func::FuncOp>(rewriter.create(state));

    // Moves the function body.
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(),
                                newFunc.getBody().end());

    // Replaces the old function.
    rewriter.replaceOp(op, newFunc);
    return success();
  }
};

struct LlvmCallToFuncCall : public OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    // Gets the callee name.
    auto callee = op.getCallee();
    if (!callee) {
      return failure();
    }

    // Checks if the callee function exists as func.func in the module.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) {
      return failure();
    }

    // Looks for a func.func with the same name.
    func::FuncOp funcOp = module.lookupSymbol<func::FuncOp>(callee.value());
    if (!funcOp) {
      return failure();
    }

    // Gets the result types from the function signature.
    auto resultTypes = funcOp.getFunctionType().getResults();


    // Converts the call to func.call.
    auto newCall = rewriter.create<func::CallOp>(
        op.getLoc(), resultTypes, callee.value(), op.getArgOperands());

    // Replaces the old call with the new one.
    // Handles both cases: calls with results and calls without results.
    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, newCall->getResults());
    }

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
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Adds DRR patterns.
    mlir::neura::llvm2neura::populateWithGenerated(patterns);
    patterns.add<LlvmConstantToNeuraConstant>(&getContext());
    // Vector operations must be registered before scalar operations
    // to ensure vector types are matched first
    patterns.add<LlvmVMulToNeuraVMul>(&getContext());
    patterns.add<LlvmVAddToNeuraVAdd>(&getContext());
    patterns.add<LlvmVFMulToNeuraVFMul>(&getContext());
    patterns.add<LlvmVFAddToNeuraVFAdd>(&getContext());
    patterns.insert<LlvmVectorReduceAddToNeuraVectorReduceAdd>(&getContext());
    // Scalar operations
    patterns.add<LlvmAddToNeuraAdd>(&getContext());
    patterns.add<LlvmAndToNeuraAnd>(&getContext());
    patterns.add<LlvmOrToNeuraOr>(&getContext());
    patterns.add<LlvmFAddToNeuraFAdd>(&getContext());
    patterns.add<LlvmFMulToNeuraFMul>(&getContext());
    patterns.add<LlvmICmpToNeuraICmp>(&getContext());
    patterns.add<LlvmFCmpToNeuraFCmp>(&getContext());
    patterns.add<LlvmGEPToNeuraGEP>(&getContext());
    patterns.add<LlvmLoadToNeuraLoad>(&getContext());
    patterns.add<LlvmStoreToNeuraStore>(&getContext());
    patterns.add<LlvmCondBrToNeuraCondBr>(&getContext());
    patterns.add<LlvmBrToNeuraBr>(&getContext());
    patterns.add<LlvmReturnToNeuraReturn>(&getContext());
    patterns.add<FuncReturnToNeuraReturn>(&getContext());
    patterns.add<LlvmFSubToNeuraFSub>(&getContext());
    patterns.add<LlvmAllocaToNeuraAlloca>(&getContext());
    patterns.add<LlvmSExtToNeuraSExt>(&getContext());
    patterns.add<LlvmZExtToNeuraZExt>(&getContext());
    patterns.add<LlvmMulToNeuraMul>(&getContext());
    patterns.add<LlvmFuncToNeuraFunc>(&getContext());
    patterns.add<LlvmCallToFuncCall>(&getContext());
    patterns.add<LlvmShlToNeuraShl>(&getContext());
    patterns.add<LlvmSDivToNeuraDiv>(&getContext());
    patterns.add<LlvmSRemToNeuraRem>(&getContext());
    patterns.add<LlvmMaxNumToNeuraFMax>(&getContext());
    patterns.add<LlvmMaximumToNeuraFMax>(&getContext());
    patterns.add<LlvmMinNumToNeuraFMin>(&getContext());
    patterns.add<LlvmMinimumToNeuraFMin>(&getContext());
    patterns.add<LlvmFDivToNeuraFDiv>(&getContext());
    patterns.add<LlvmFPToSIToNeuraCast>(&getContext());
    patterns.add<LlvmFMulAddToNeuraFMulFAdd>(&getContext());
    patterns.add<LlvmSelectToNeuraSel>(&getContext());
    patterns.add<LlvmMemsetToNeuraOps>(&getContext());
    patterns.add<LlvmFNegToNeuraFSub>(&getContext());
    patterns.add<LlvmSubToNeuraSub>(&getContext());
    patterns.add<LlvmTruncToNeuraCast>(&getContext());
    patterns.add<LlvmUDivToNeuraDiv>(&getContext());
    patterns.add<LlvmURemToNeuraRem>(&getContext());
    patterns.add<LlvmSMaxToNeuraSMax>(&getContext());
    // TODO: Add more LLVM to Neura conversion patterns as needed
    // patterns.add<LlvmXOrToNeuraOr>(&getContext());     // TODO: Used in ADPCM coder + FFT kernels
    // patterns.add<LlvmAndToNeuraMul>(&getContext());    // TODO: Used in ADPCM coder + MVT kernels  
    // patterns.add<LlvmAllocaToNeuraOps>(&getContext()); // TODO: Used in DTW kernel
    // TODO: Fix right shift implementations - current implementations are incorrect
    // patterns.add<LlvmLShrToNeuraShl>(&getContext());  // TODO: Used in ADPCM coder/decoder + FFT kernels
    // patterns.add<LlvmAShrToNeuraAShr>(&getContext()); // TODO: Used in ADPCM coder/decoder kernels
    // patterns.add<LlvmAbsToNeuraAbs>(&getContext());   // TODO: Used in ADPCM coder kernel


    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();

    // Performs function-level conversions.
    if (failed(applyPatternsGreedily(module_op, frozen))) {
      signalPassFailure();
      return;
    }

    // Performs operation-level conversions.
    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](FunctionOpInterface func) {
      if (func->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target =
            func->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (target && target.getValue() == mlir::accel::kNeuraTarget) {
          for (Region &region : func->getRegions()) {
            if (failed(applyPatternsGreedily(region, frozen))) {
              signalPassFailure();
            }
          }
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerLlvmToNeuraPass() {
  return std::make_unique<LowerLlvmToNeuraPass>();
}
