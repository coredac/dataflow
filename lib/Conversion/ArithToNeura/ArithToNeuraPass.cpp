#include "Common/AcceleratorAttrs.h"
#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

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

namespace {

struct ArithConstantToNeuraConstant
    : public OpRewritePattern<mlir::arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    // Converts arith constant to Neura constant.
    Type result_type = op.getType();
    Attribute value = op.getValue();
    // Optional predicate parameter can be null.
    rewriter.replaceOpWithNewOp<neura::ConstantOp>(op, result_type, value,
                                                   rewriter.getBoolAttr(true));
    return success();
  }
};

struct ArithAddIToNeuraAdd : public OpRewritePattern<mlir::arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::AddOp>(op, result_type, lhs, rhs,
                                              nullptr);
    return success();
  }
};

struct ArithFAddToNeuraFAdd : public OpRewritePattern<mlir::arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::FAddOp>(op, result_type, lhs, rhs,
                                               nullptr);
    return success();
  }
};

struct ArithSubIToNeuraSub : public OpRewritePattern<mlir::arith::SubIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::SubOp>(op, result_type, lhs, rhs,
                                              nullptr);
    return success();
  }
};

struct ArithSubFToNeuraFSub : public OpRewritePattern<mlir::arith::SubFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::FSubOp>(op, result_type, lhs, rhs,
                                               nullptr);
    return success();
  }
};

struct ArithMulIToNeuraMul : public OpRewritePattern<mlir::arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::MulOp>(op, result_type, lhs, rhs,
                                              nullptr);
    return success();
  }
};

struct ArithMulFToNeuraFMul : public OpRewritePattern<mlir::arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::FMulOp>(op, result_type, lhs, rhs,
                                               nullptr);
    return success();
  }
};

struct ArithDivSIToNeuraDiv : public OpRewritePattern<mlir::arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    // Converts arith DivSIOp to Neura DivOp.
    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::DivOp>(op, result_type, lhs, rhs,
                                              nullptr);
    return success();
  }
};

struct ArithFDivToNeuraFDiv : public OpRewritePattern<mlir::arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();

    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::FDivOp>(op, result_type, lhs, rhs,
                                               nullptr);
    return success();
  }
};

struct ArithRemSIToNeuraOp : public OpRewritePattern<mlir::arith::RemSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    Location loc = op.getLoc();
    // Converts arith RemSIOp to basic Neura Op.
    // Optional predicate: default to null.
    Value div =
        rewriter.create<neura::DivOp>(loc, result_type, lhs, rhs, nullptr);
    Value mul =
        rewriter.create<neura::MulOp>(loc, result_type, rhs, div, nullptr);
    Value rem =
        rewriter.create<neura::SubOp>(loc, result_type, lhs, mul, nullptr);

    rewriter.replaceOp(op, rem);
    return success();
  }
};

struct ArithCmpiToNeuraICmp : public OpRewritePattern<mlir::arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type result_type = op.getType();
    arith::CmpIPredicate arith_cmp_type = op.getPredicate();
    StringRef cmp_type;
    switch (arith_cmp_type) {
    case arith::CmpIPredicate::eq:
      cmp_type = "eq"; // ==
      break;
    case arith::CmpIPredicate::ne:
      cmp_type = "ne"; // !=
      break;
    case arith::CmpIPredicate::slt:
      cmp_type = "slt"; // <
      break;
    case arith::CmpIPredicate::sle:
      cmp_type = "sle"; // <=
      break;
    case arith::CmpIPredicate::sgt:
      cmp_type = "sgt"; // >
      break;
    case arith::CmpIPredicate::sge:
      cmp_type = "sge"; // >=
      break;
    case arith::CmpIPredicate::ult:
      cmp_type = "ult"; // unsigned <
      break;
    case arith::CmpIPredicate::ule:
      cmp_type = "ule"; // unsigned <=
      break;
    case arith::CmpIPredicate::ugt:
      cmp_type = "ugt"; // unsigned >
      break;
    case arith::CmpIPredicate::uge:
      cmp_type = "uge"; // unsigned >=
      break;
    default:
      return rewriter.notifyMatchFailure(op, "Unsupported arith CmpIOp type");
    }

    // Converts arith CmpIOp to Neura ICmpOp.
    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::ICmpOp>(
        op, result_type, lhs, rhs, nullptr, rewriter.getStringAttr(cmp_type));
    return success();
  }
};

struct ArithSelectToNeuraSel : public OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value condition = op.getCondition();
    Value true_value = op.getTrueValue();
    Value false_value = op.getFalseValue();
    Type result_type = op.getType();

    // Converts arith SelectOp to Neura SelOp.
    rewriter.replaceOpWithNewOp<neura::SelOp>(op, result_type, true_value,
                                              false_value, condition);
    return success();
  }
};

struct ArithExtUIToNeuraCast : public OpRewritePattern<mlir::arith::ExtUIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtUIOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    Type result_type = op.getType();

    // Converts arith ExtUIOp to Neura cast operation.
    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::CastOp>(
        op, result_type, input, rewriter.getStringAttr("extui"), nullptr);
    return success();
  }
};

struct ArithExtfToNeuraCast : public OpRewritePattern<mlir::arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    Type result_type = op.getType();

    // Converts arith ExtFOp to Neura cast operation.
    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::CastOp>(
        op, result_type, input, rewriter.getStringAttr("extf"), nullptr);
    return success();
  }
};

struct ArithIndexCastToNeuraCast
    : public OpRewritePattern<mlir::arith::IndexCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getIn();
    Type result_type = op.getType();
    Type in_type = input.getType();
    StringRef cast_string;

    // The isa<IntegerType> check is generic and handles any integer bit
    // width (e.g., i32, i64).
    if (in_type.isIndex() && isa<IntegerType>(result_type)) {
      cast_string = "index_to_int";
    } else if (isa<IntegerType>(in_type) && result_type.isIndex()) {
      cast_string = "int_to_index";
    } else {
      return rewriter.notifyMatchFailure(op, "index_cast");
    }

    // Converts arith IndexCastOp to Neura cast operation.
    // Optional predicate: default to null.
    rewriter.replaceOpWithNewOp<neura::CastOp>(
        op, result_type, input, rewriter.getStringAttr(cast_string), nullptr);
    return success();
  }
};

struct LowerArithToNeuraPass
    : public PassWrapper<LowerArithToNeuraPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerArithToNeuraPass)

  StringRef getArgument() const override { return "lower-arith-to-neura"; }
  StringRef getDescription() const override {
    return "Lower arith dialect operations to Neura dialect operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    MLIRContext *context = &getContext();
    module_op.walk([&](func::FuncOp func_op) {
      if (func_op->hasAttr(mlir::accel::kAcceleratorAttr)) {
        auto target =
            func_op->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
        if (target && target.getValue() == mlir::accel::kNeuraTarget) {
          RewritePatternSet patterns(&getContext());
          mlir::neura::arith2neura::populateWithGenerated(patterns);
          patterns.add<
              ArithFAddToNeuraFAdd, ArithConstantToNeuraConstant,
              ArithAddIToNeuraAdd, ArithCmpiToNeuraICmp, ArithSelectToNeuraSel,
              ArithExtUIToNeuraCast, ArithIndexCastToNeuraCast,
              ArithFDivToNeuraFDiv, ArithExtfToNeuraCast, ArithMulFToNeuraFMul,
              ArithSubIToNeuraSub, ArithSubFToNeuraFSub, ArithMulIToNeuraMul,
              ArithDivSIToNeuraDiv, ArithRemSIToNeuraOp>(context);
          if (failed(
                  applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerArithToNeuraPass() {
  return std::make_unique<LowerArithToNeuraPass>();
}
