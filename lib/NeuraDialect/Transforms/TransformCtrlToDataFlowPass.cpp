#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

#define GEN_PASS_DEF_TransformCtrlToDataFlow
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct TransformCtrlToDataFlowPass : public PassWrapper<TransformCtrlToDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformCtrlToDataFlowPass)

  StringRef getArgument() const override { return "transform-ctrl-to-data-flow"; }
  StringRef getDescription() const override {
    return "Flatten control flow into predicated linear SSA for Neura dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      if (!func->hasAttr(mlir::accel::kAcceleratorAttr))
        return;
      auto target = func->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
      if (!target || target.getValue() != mlir::accel::kNeuraTarget)
        return;

      Block &entryBlock = func.getBody().front();
      OpBuilder builder(func.getContext());

      // Collect cond_br terminator from entry block (assume 2-way for now)
      auto *term = entryBlock.getTerminator();
      auto condBr = dyn_cast_or_null<neura::CondBr>(term);
      if (!condBr) return;

      Location loc = condBr.getLoc();
      Value cond = condBr.getCondition();
      builder.setInsertionPointToEnd(&entryBlock);
      auto notCond = builder.create<neura::NotOp>(loc, cond.getType(), cond);

      Block *trueBlock = condBr.getTrueDest();
      Block *falseBlock = condBr.getFalseDest();

      // Clone ops from true and false blocks into entry block
      SmallVector<Value> trueResults, falseResults;

      for (Operation &op : llvm::make_early_inc_range(*trueBlock)) {
        if (op.hasTrait<OpTrait::IsTerminator>()) continue;
        builder.setInsertionPointToEnd(&entryBlock);
        Operation *cloned = builder.clone(op);
        cloned->insertOperands(cloned->getNumOperands(), cond);
        trueResults.push_back(cloned->getResult(0));
      }

      for (Operation &op : llvm::make_early_inc_range(*falseBlock)) {
        if (op.hasTrait<OpTrait::IsTerminator>()) continue;
        builder.setInsertionPointToEnd(&entryBlock);
        Operation *cloned = builder.clone(op);
        cloned->insertOperands(cloned->getNumOperands(), notCond.getResult());
        falseResults.push_back(cloned->getResult(0));
      }

      // Merge block arguments with neura.sel
      builder.setInsertionPointToEnd(&entryBlock);
      Block *mergeBlock = condBr.getTrueDest()->getSuccessors()[0]; // assumes shared ^bb3
      assert(mergeBlock->getNumArguments() == trueResults.size());

      for (auto [idx, arg] : llvm::enumerate(mergeBlock->getArguments())) {
        auto sel = builder.create<neura::SelOp>(
            loc, arg.getType(), trueResults[idx], falseResults[idx], cond);
        arg.replaceAllUsesWith(sel.getResult());
      }

      // Insert return
      Operation *returnOp = &mergeBlock->back();
      if (returnOp->getNumOperands() == 1) {
        builder.setInsertionPointToEnd(&entryBlock);
        builder.create<func::ReturnOp>(loc, returnOp->getOperand(0));
      }

      // Cleanup: remove old blocks and terminator
      condBr.erase();
      for (auto it = ++func.getBody().begin(); it != func.getBody().end();) {
        it = func.getBody().getBlocks().erase(it);
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura
