#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
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

  // By default, we assume the operation is not commutative.
  // If the operation is commutative, we can extend this pattern to support
  // constant folding on the left-hand side operand as well.
  virtual bool isCommutative() const { return false; }

  virtual Operation *
  createOpWithFusedRhsConstant(OpType op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const = 0;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("rhs_value")) {
      // Already fused with a constant on the right-hand side.
      return failure();
    }

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    bool lhs_is_const = isOriginConstantOp(lhs);
    bool rhs_is_const = rhs && isOriginConstantOp(rhs);

    if (rhs_is_const) {
      auto constant_op = dyn_cast<neura::ConstantOp>(rhs.getDefiningOp());

      Attribute rhs_value = getOriginConstantValue(rhs);
      Operation *fused_op =
          createOpWithFusedRhsConstant(op, lhs, rhs_value, rewriter);

      rewriter.replaceOp(op, fused_op->getResults());
      if (constant_op->use_empty()) {
        rewriter.eraseOp(constant_op);
      }
      return success();
    }

    if (lhs_is_const && !rhs_is_const && isCommutative()) {
      auto constant_op = dyn_cast<neura::ConstantOp>(lhs.getDefiningOp());

      Attribute lhs_value = getOriginConstantValue(lhs);
      Operation *fused_op =
          createOpWithFusedRhsConstant(op, rhs, lhs_value, rewriter);

      rewriter.replaceOp(op, fused_op->getResults());
      if (constant_op->use_empty()) {
        rewriter.eraseOp(constant_op);
      }
      return success();
    }

    return failure();
  }
};

struct FuseAddRhsConstantPattern : public FuseRhsConstantPattern<neura::AddOp> {
  using FuseRhsConstantPattern<neura::AddOp>::FuseRhsConstantPattern;

  bool isCommutative() const override { return true; }

  Operation *
  createOpWithFusedRhsConstant(neura::AddOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::AddOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseSubRhsConstantPattern : public FuseRhsConstantPattern<neura::SubOp> {
  using FuseRhsConstantPattern<neura::SubOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::SubOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::SubOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseMulRhsConstantPattern : public FuseRhsConstantPattern<neura::MulOp> {
  using FuseRhsConstantPattern<neura::MulOp>::FuseRhsConstantPattern;

  bool isCommutative() const override { return true; }

  Operation *
  createOpWithFusedRhsConstant(neura::MulOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::MulOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseICmpRhsConstantPattern
    : public FuseRhsConstantPattern<neura::ICmpOp> {
  using FuseRhsConstantPattern<neura::ICmpOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::ICmpOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::ICmpOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr, op.getCmpType());
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseFAddRhsConstantPattern
    : public FuseRhsConstantPattern<neura::FAddOp> {
  using FuseRhsConstantPattern<neura::FAddOp>::FuseRhsConstantPattern;

  bool isCommutative() const override { return true; }

  Operation *
  createOpWithFusedRhsConstant(neura::FAddOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::FAddOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseFSubRhsConstantPattern
    : public FuseRhsConstantPattern<neura::FSubOp> {
  using FuseRhsConstantPattern<neura::FSubOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::FSubOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::FSubOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseFMulRhsConstantPattern
    : public FuseRhsConstantPattern<neura::FMulOp> {
  using FuseRhsConstantPattern<neura::FMulOp>::FuseRhsConstantPattern;

  bool isCommutative() const override { return true; }

  Operation *
  createOpWithFusedRhsConstant(neura::FMulOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::FMulOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseFMaxRhsConstantPattern
    : public FuseRhsConstantPattern<neura::FMaxOp> {
  using FuseRhsConstantPattern<neura::FMaxOp>::FuseRhsConstantPattern;

  bool isCommutative() const override { return true; }

  Operation *
  createOpWithFusedRhsConstant(neura::FMaxOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::FMaxOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr, op.getNanSemantic());
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseFMinRhsConstantPattern
    : public FuseRhsConstantPattern<neura::FMinOp> {
  using FuseRhsConstantPattern<neura::FMinOp>::FuseRhsConstantPattern;

  bool isCommutative() const override { return true; }

  Operation *
  createOpWithFusedRhsConstant(neura::FMinOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::FMinOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr, op.getNanSemantic());
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseDivRhsConstantPattern : public FuseRhsConstantPattern<neura::DivOp> {
  using FuseRhsConstantPattern<neura::DivOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::DivOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::DivOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

struct FuseRemRhsConstantPattern : public FuseRhsConstantPattern<neura::RemOp> {
  using FuseRhsConstantPattern<neura::RemOp>::FuseRhsConstantPattern;

  Operation *
  createOpWithFusedRhsConstant(neura::RemOp op, Value non_const_operand,
                               Attribute rhs_value,
                               PatternRewriter &rewriter) const override {
    auto fused_op = rewriter.create<neura::RemOp>(
        op.getLoc(), op.getResult().getType(), non_const_operand,
        /*rhs=*/nullptr);
    addConstantAttribute(fused_op, "rhs_value", rhs_value);
    return fused_op;
  }
};

// =========================================
// FuseGepBaseConstantPattern
// Folds constant base pointer for GEP operation.
// =========================================
struct FuseGepBaseConstantPattern : public OpRewritePattern<neura::GEP> {
  using OpRewritePattern<neura::GEP>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::GEP gep_op,
                                PatternRewriter &rewriter) const override {
    Value base = gep_op.getBase();
    
    // Checks if base exists and is a constant.
    if (!base || !isOriginConstantOp(base)) {
      return failure();
    }

    auto constant_op = dyn_cast<neura::ConstantOp>(base.getDefiningOp());
    Attribute base_value = getOriginConstantValue(base);

    // Gets all indices (everything after base).
    SmallVector<Value> indices;
    for (Value operand : gep_op.getIndices()) {
      indices.push_back(operand);
    }

    // Creates new GEP with no base but with lhs_value attribute.
    auto fused_gep = rewriter.create<neura::GEP>(
        gep_op.getLoc(), 
        gep_op.getResult().getType(),
        /*base=*/nullptr,
        indices);
    // TODO: Gather all the attribute -- https://github.com/coredac/dataflow/issues/145
    addConstantAttribute(fused_gep, "lhs_value", base_value);

    // Replaces the original GEP.
    rewriter.replaceOp(gep_op, fused_gep);
    
    // Cleans up constant if no longer used.
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }
    
    return success();
  }
};

// =========================================
// FuseStoreAddrConstantPattern
// Folds constant destination pointer for Store operation.
// =========================================
struct FuseStoreAddrConstantPattern : public OpRewritePattern<neura::StoreOp> {
  using OpRewritePattern<neura::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::StoreOp store_op,
                                PatternRewriter &rewriter) const override {
    Value addr = store_op.getAddr();
    
    // Checks if address exists and is a constant.
    if (!addr || !isOriginConstantOp(addr)) {
      return failure();
    }

    auto constant_op = dyn_cast<neura::ConstantOp>(addr.getDefiningOp());
    Attribute addr_value = getOriginConstantValue(addr);

    // Creates new Store with no addr but with rhs_value attribute.
    auto fused_store = rewriter.create<neura::StoreOp>(
        store_op.getLoc(),
        store_op.getValue(),  // Keeps the value operand.
        /*addr=*/nullptr);    // Removes addr operand.
    addConstantAttribute(fused_store, "rhs_value", addr_value);

    // Replaces the original Store.
    rewriter.replaceOp(store_op, fused_store);
    
    // Cleans up constant if no longer used.
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }
    
    return success();
  }
};

// =========================================
// FuseLoadIndexedBaseConstantPattern
// Folds constant base pointer for LoadIndexed operation.
// =========================================
struct FuseLoadIndexedBaseConstantPattern
    : public OpRewritePattern<neura::LoadIndexedOp> {
  using OpRewritePattern<neura::LoadIndexedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::LoadIndexedOp load_indexed_op,
                                PatternRewriter &rewriter) const override {
    Value base = load_indexed_op.getBase();
    
    // Checks if base exists and is a constant.
    if (!base || !isOriginConstantOp(base)) {
      return failure();
    }

    auto constant_op = dyn_cast<neura::ConstantOp>(base.getDefiningOp());
    Attribute base_const_value = getOriginConstantValue(base);

    // Gets all indices.
    SmallVector<Value> indices;
    for (Value idx : load_indexed_op.getIndices()) {
      indices.push_back(idx);
    }

    // Creates new LoadIndexed with no base but with lhs_value attribute.
    auto fused_load_indexed = rewriter.create<neura::LoadIndexedOp>(
        load_indexed_op.getLoc(),
        load_indexed_op.getResult().getType(),
        /*base=*/nullptr,
        indices);
    addConstantAttribute(fused_load_indexed, "lhs_value", base_const_value);

    // Replaces the original LoadIndexed.
    rewriter.replaceOp(load_indexed_op, fused_load_indexed);
    
    // Cleans up constant if no longer used.
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }
    
    return success();
  }
};

// =========================================
// FuseStoreIndexedBaseConstantPattern
// Folds constant base pointer for StoreIndexed operation.
// =========================================
struct FuseStoreIndexedBaseConstantPattern
    : public OpRewritePattern<neura::StoreIndexedOp> {
  using OpRewritePattern<neura::StoreIndexedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(neura::StoreIndexedOp store_indexed_op,
                                PatternRewriter &rewriter) const override {
    Value base = store_indexed_op.getBase();
    
    // Checks if base exists and is a constant.
    if (!base || !isOriginConstantOp(base)) {
      return failure();
    }

    auto constant_op = dyn_cast<neura::ConstantOp>(base.getDefiningOp());
    Attribute base_const_value = getOriginConstantValue(base);

    // Gets all indices.
    SmallVector<Value> indices;
    for (Value idx : store_indexed_op.getIndices()) {
      indices.push_back(idx);
    }

    // Creates new StoreIndexed with no base but with rhs_value attribute.
    auto fused_store_indexed = rewriter.create<neura::StoreIndexedOp>(
        store_indexed_op.getLoc(),
        store_indexed_op.getValue(),  // Keeps the value operand.
        /*base=*/nullptr,
        indices);
    addConstantAttribute(fused_store_indexed, "rhs_value", base_const_value);

    // Replaces the original StoreIndexed.
    rewriter.replaceOp(store_indexed_op, fused_store_indexed);
    
    // Cleans up constant if no longer used.
    if (constant_op->use_empty()) {
      rewriter.eraseOp(constant_op);
    }
    
    return success();
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
    patterns.add<FuseFAddRhsConstantPattern>(&getContext());
    patterns.add<FuseFSubRhsConstantPattern>(&getContext());
    patterns.add<FuseFMulRhsConstantPattern>(&getContext());
    patterns.add<FuseFMaxRhsConstantPattern>(&getContext());
    patterns.add<FuseFMinRhsConstantPattern>(&getContext());
    patterns.add<FuseDivRhsConstantPattern>(&getContext());
    patterns.add<FuseRemRhsConstantPattern>(&getContext());

    patterns.add<FuseConstantAndGrantPattern>(&getContext());
    patterns.add<FuseGepBaseConstantPattern>(&getContext());
    patterns.add<FuseStoreAddrConstantPattern>(&getContext());
    patterns.add<FuseLoadIndexedBaseConstantPattern>(&getContext());
    patterns.add<FuseStoreIndexedBaseConstantPattern>(&getContext());
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
