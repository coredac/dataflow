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

// Processes a block recursively, cloning its operations into the entry block.
void processBlockRecursively(Block *block, Block &entry_block, Value predicate, OpBuilder &builder,
                           SmallVector<Value> &results, DenseSet<Block *> &visited_blocks,
                           DenseMap<BlockArgument, Value> &arg_mapping,
                           DenseMap<Value, Value> &value_mapping) {
  // Check if the block has already been visited
  if (visited_blocks.contains(block)) {
    llvm::errs() << "Skipping already visited block:\n";
    block->dump();
    return;
  }

  // Mark the block as visited
  visited_blocks.insert(block);

  llvm::errs() << "Processing block:\n";
  block->dump();

  // Handle block arguments first
  for (BlockArgument arg : block->getArguments()) {
    llvm::errs() << "Processing block argument: " << arg << "\n";
    
    // Check if we already have a mapping for this argument
    if (auto mapped = arg_mapping.lookup(arg)) {
      llvm::errs() << "Found existing mapping for argument\n";
      continue;
    }

    builder.setInsertionPointToEnd(&entry_block);
    // Create a new constant operation with zero value and true predicate
    OperationState state(arg.getLoc(), neura::ConstantOp::getOperationName());
    state.addAttribute("value", builder.getZeroAttr(arg.getType()));
    state.addAttribute("predicate", builder.getBoolAttr(true));
    state.addTypes(arg.getType());
    Value false_val = builder.create(state)->getResult(0);

    llvm::errs() << "Creating false_val: \n";
    false_val.dump();
    auto sel = builder.create<neura::SelOp>(
        arg.getLoc(), arg.getType(), arg, false_val, predicate);

    llvm::errs() << "Created sel operation for argument:\n";
    sel->dump();

    // Store mapping
    arg_mapping.try_emplace(arg, sel.getResult());
    value_mapping[arg] = sel.getResult();
    results.push_back(sel.getResult());
  }

  // Process operations
  SmallVector<Operation *> ops_to_process;
  for (Operation &op : *block) {
    ops_to_process.push_back(&op);
  }

  for (Operation *op : ops_to_process) {
    llvm::errs() << "Processing operation:\n";
    op->dump();

    if (op->hasTrait<OpTrait::IsTerminator>()) {
      if (auto br = dyn_cast<neura::Br>(op)) {
        llvm::errs() << "Found unconditional branch\n";
        for (Value operand : br.getOperands()) {
          if (auto mapped = value_mapping.lookup(operand)) {
            results.push_back(mapped);
          } else {
            results.push_back(operand);
          }
        }
      } else if (auto cond_br = dyn_cast<neura::CondBr>(op)) {
        llvm::errs() << "Found conditional branch\n";
        Value cond = cond_br.getCondition();
        auto not_cond = builder.create<neura::NotOp>(cond_br.getLoc(), cond.getType(), cond);

        SmallVector<Value> true_results, false_results;
        processBlockRecursively(cond_br.getTrueDest(), entry_block, cond, 
                              builder, true_results, visited_blocks, arg_mapping, value_mapping);
        processBlockRecursively(cond_br.getFalseDest(), entry_block, not_cond.getResult(), 
                              builder, false_results, visited_blocks, arg_mapping, value_mapping);

        builder.setInsertionPointToEnd(&entry_block);
        for (auto [true_result, false_result] : llvm::zip(true_results, false_results)) {
          auto sel = builder.create<neura::SelOp>(
              op->getLoc(), true_result.getType(), true_result, false_result, cond);
          value_mapping[sel.getResult()] = sel.getResult();
          results.push_back(sel.getResult());
        }
      } else if (auto ret = dyn_cast<neura::ReturnOp>(op)) {
        llvm::errs() << "Found Return\n";
        for (Value operand : ret.getOperands()) {
          if (auto mapped = value_mapping.lookup(operand)) {
            results.push_back(mapped);
          } else {
            results.push_back(operand);
          }
        }
      } else {
        // Handle other terminators if needed
        llvm::errs() << "Found unexpected terminator operation:\n";
        op->dump();
        assert(false && "Unexpected terminator operation in block");
      }
    }

    builder.setInsertionPointToEnd(&entry_block);
    Operation *cloned_op = builder.clone(*op);

    // Replace operands with mapped values
    for (unsigned i = 0; i < cloned_op->getNumOperands(); ++i) {
      Value operand = cloned_op->getOperand(i);
      if (auto mapped = value_mapping.lookup(operand)) {
        cloned_op->setOperand(i, mapped);
      }
    }

    if (!cloned_op->hasTrait<OpTrait::IsTerminator>()) {
      cloned_op->insertOperands(cloned_op->getNumOperands(), predicate);
    }

    // Store mappings and results
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      Value orig_result = op->getResult(i);
      Value new_result = cloned_op->getResult(i);
      value_mapping[orig_result] = new_result;
      results.push_back(new_result);
    }
  }
  llvm::errs() << "[cheng] after processing entry_block:\n";
  entry_block.dump();
}

namespace {
struct TransformCtrlToDataFlowPass 
    : public PassWrapper<TransformCtrlToDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformCtrlToDataFlowPass)

  StringRef getArgument() const override { return "transform-ctrl-to-data-flow"; }
  StringRef getDescription() const override {
    return "Flattens control flow into predicated linear SSA for Neura dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      llvm::errs() << "Processing function: ";
      func.dump();

      if (!func->hasAttr(mlir::accel::kAcceleratorAttr))
        return;

      auto target = func->getAttrOfType<StringAttr>(mlir::accel::kAcceleratorAttr);
      if (!target || target.getValue() != mlir::accel::kNeuraTarget)
        return;

      Block &entry_block = func.getBody().front();
      llvm::errs() << "Entry block before processing:\n";
      entry_block.dump();

      OpBuilder builder(&entry_block, entry_block.begin());

      // Check for terminator
      Operation *terminator = nullptr;
      if (!entry_block.empty()) {
        terminator = &entry_block.back();
      }

      auto cond_br = dyn_cast_or_null<neura::CondBr>(terminator);
      if (!cond_br) {
        llvm::errs() << "No conditional branch found in entry block\n";
        return;
      }

      // Get condition and create not condition
      Location loc = cond_br.getLoc();
      Value cond = cond_br.getCondition();
      builder.setInsertionPoint(cond_br);
      auto not_cond = builder.create<neura::NotOp>(loc, cond.getType(), cond);

      // Process branches
      DenseMap<BlockArgument, Value> arg_mapping;
      DenseMap<Value, Value> value_mapping;
      DenseSet<Block *> visited_blocks;
      SmallVector<Value> true_results, false_results;

      processBlockRecursively(cond_br.getTrueDest(), entry_block, cond,
                            builder, true_results, visited_blocks, arg_mapping, value_mapping);
      processBlockRecursively(cond_br.getFalseDest(), entry_block, not_cond.getResult(),
                            builder, false_results, visited_blocks, arg_mapping, value_mapping);

      llvm::errs() << "Entry block after processing:\n";
      entry_block.dump();

      // Create final return operation
      if (!true_results.empty() && !false_results.empty()) {
        builder.setInsertionPoint(cond_br);
        auto sel = builder.create<neura::SelOp>(
            loc, true_results[0].getType(), true_results[0], false_results[0], cond);
        builder.create<func::ReturnOp>(loc, sel.getResult());
      }

      // Replace all uses with mapped values
      for (auto &[orig, mapped] : value_mapping) {
        orig.replaceAllUsesWith(mapped);
      }

      // Now erase the conditional branch
      cond_br->erase();

      // Finally erase all other blocks
      SmallVector<Block *> blocks_to_erase;
      for (Block &block : llvm::make_early_inc_range(func.getBody())) {
        if (&block != &entry_block) {
          blocks_to_erase.push_back(&block);
        }
      }

      for (Block *block : blocks_to_erase) {
        block->dropAllReferences();
        block->erase();
      }

      llvm::errs() << "Function after transformation:\n";
      func.dump();
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura