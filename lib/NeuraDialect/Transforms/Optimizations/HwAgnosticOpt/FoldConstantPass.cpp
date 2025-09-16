#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace mlir;

#define GEN_PASS_DEF_FOLDCONSTANT
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// =========================================
// FuseConstantAndGrantPattern
// Valid only after transform-ctrl-to-data-flow pass.
// =========================================
struct FuseConstantAndGrantPattern
    : public OpRewritePattern<neura::ConstantOp> {
  using OpRewritePattern<neura::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::ConstantOp constant_op,
                                PatternRewriter &rewriter) const override {
    bool made_change = false;

    // Checks if the constant operation is used by a grant_once or grant_always
    // operation.
    for (auto user : constant_op->getUsers()) {
      llvm::errs() << "Checking use: " << *user << "\n";
      if (isa<neura::GrantOnceOp>(user) || isa<neura::GrantAlwaysOp>(user)) {
        if (neura::GrantOnceOp grant_once_op =
                dyn_cast<neura::GrantOnceOp>(user)) {
          auto new_grant_once_op = rewriter.create<neura::GrantOnceOp>(
              grant_once_op.getLoc(), grant_once_op.getResult().getType(),
              /*value=*/nullptr, constant_op->getAttr("value"));
          // Replaces the original constant operation with the new one.
          rewriter.replaceOp(grant_once_op, new_grant_once_op);
          made_change = true;
        } else if (neura::GrantAlwaysOp grant_always_op =
                       dyn_cast<neura::GrantAlwaysOp>(user)) {
          auto new_grant_always_op = rewriter.create<neura::GrantAlwaysOp>(
              grant_always_op.getLoc(), grant_always_op.getResult().getType(),
              /*value=*/nullptr, constant_op->getAttr("value"));
          // Replaces the original constant operation with the new one.
          rewriter.replaceOp(grant_always_op, new_grant_always_op);
          made_change = true;
        }
      }
    }

    if (constant_op->use_empty()) {
      // If the constant operation has no users, it can be removed.
      rewriter.eraseOp(constant_op);
      made_change = true;
    }

    return success(made_change);
  }
};

// =========================================
// FoldConstantPass
// Valid before transform-ctrl-to-data-flow pass.
// =========================================
bool isOriginConstantOp(Value value) {
  Operation *def_op = value.getDefiningOp();
  if (!def_op || !isa<neura::ConstantOp>(def_op)) {
    return false;
  }

  // Checks if the result type is the original type or the predicated type.
  Type result_type = value.getType();
  if (isa<neura::PredicatedValue>(result_type)) {
    return false;
  }

  return true;
}

Attribute getOriginConstantValue(Value value) {
  neura::ConstantOp constant_op =
      dyn_cast<neura::ConstantOp>(value.getDefiningOp());
  return constant_op->getAttr("value");
}

void addConstantAttribute(Operation *op, StringRef attr_name,
                          Attribute const_value) {
  op->setAttr(attr_name, const_value);
}

// A template pattern to fuse binary operations with a constant on the
// right-hand side operand.
template <typename OpType>
struct FuseRhsConstantPattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  virtual Operation *
  createOpWithFusedRhsConstant(OpType op, Attribute rhs_const_value,
                               PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (isOriginConstantOp(lhs)) {
      assert(false && "LHS constant folding not implemented yet.");
      return failure();
    }

    if (!rhs || !isOriginConstantOp(rhs)) {
      return failure();
    }

    auto constant_op = dyn_cast<neura::ConstantOp>(rhs.getDefiningOp());

    Attribute rhs_const_value = getOriginConstantValue(rhs);
    Operation *fused_op =
        createOpWithFusedRhsConstant(op, rhs_const_value, rewriter);

    rewriter.replaceOp(op, fused_op->getResults());
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }
    return success();
  }
};

struct FuseAddRhsConstantPattern : public FuseRhsConstantPattern<neura::AddOp> {
  using FuseRhsConstantPattern<neura::AddOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::AddOp op, Attribute rhs_const_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::AddOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(),
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_const_value", rhs_const_value);
    return fused_op;
  }
};

struct FuseSubRhsConstantPattern : public FuseRhsConstantPattern<neura::SubOp> {
  using FuseRhsConstantPattern<neura::SubOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::SubOp op, Attribute rhs_const_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::SubOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(),
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_const_value", rhs_const_value);
    return fused_op;
  }
};

struct FuseMulRhsConstantPattern : public FuseRhsConstantPattern<neura::MulOp> {
  using FuseRhsConstantPattern<neura::MulOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::MulOp op, Attribute rhs_const_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::MulOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(),
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_const_value", rhs_const_value);
    return fused_op;
  }
};

struct FuseICmpRhsConstantPattern
    : public FuseRhsConstantPattern<neura::ICmpOp> {
  using FuseRhsConstantPattern<neura::ICmpOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::ICmpOp op, Attribute rhs_const_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::ICmpOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(),
        /*rhs=*/nullptr, op.getCmpType());
    addConstantAttribute(fused_op, "rhs_const_value", rhs_const_value);
    return fused_op;
  }
};

struct FuseFAddRhsConstantPattern
    : public FuseRhsConstantPattern<neura::FAddOp> {
  using FuseRhsConstantPattern<neura::FAddOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::FAddOp op, Attribute rhs_const_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::FAddOp>(
        op.getLoc(), op.getResult().getType(), op.getLhs(),
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_const_value", rhs_const_value);
    return fused_op;
  }
};

// =========================================
// FoldConstantPass Implementation
// =========================================
struct FoldConstantPass
    : public PassWrapper<FoldConstantPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConstantPass)

  StringRef getArgument() const override { return "fold-constant"; }
  StringRef getDescription() const override {
    return "Fold constant operations.";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    RewritePatternSet patterns(&getContext());

    patterns.add<FuseAddRhsConstantPattern>(&getContext());
    patterns.add<FuseSubRhsConstantPattern>(&getContext());
    patterns.add<FuseMulRhsConstantPattern>(&getContext());
    patterns.add<FuseICmpRhsConstantPattern>(&getContext());

    patterns.add<FuseConstantAndGrantPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createFoldConstantPass() {
  return std::make_unique<FoldConstantPass>();
}
} // namespace mlir::neura
