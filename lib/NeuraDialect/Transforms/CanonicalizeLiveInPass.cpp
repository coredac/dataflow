#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include <cassert>
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZELIVEIN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
LogicalResult promoteFunctionArgsToConstants(Region &region) {
  if (region.empty()) {
    return success();
  }

  Block &entry_block = region.front();
  OpBuilder builder(&entry_block, entry_block.begin());

  // Collects all function arguments.
  SmallVector<BlockArgument, 4> args(entry_block.getArguments().begin(),
                                     entry_block.getArguments().end());

  // Creates a constant operation for each function argument.
  for (auto [idx, arg] : llvm::enumerate(args)) {
    // For constant operation, the default predicate is true.
    auto const_op = builder.create<neura::ConstantOp>(
        arg.getLoc(), arg.getType(),
        builder.getStringAttr("\%arg" + std::to_string(idx)),
        builder.getBoolAttr(true));
    arg.replaceAllUsesWith(const_op.getResult());
  }

  return success();
}

LogicalResult promoteLiveInValuesToBlockArgs(Region &region) {
  if (region.empty()) {
    return success();
  }
  // Collects direct live-in values for each block in the region.
  // Without considering the transitive dependencies.
  DenseMap<Block *, SetVector<Value>> direct_live_ins;

  Block &entry_block = region.front();
  // Initializes the direct live-ins for each block.
  for (Block &block : region.getBlocks()) {
    if (&block == &entry_block) {
      continue;
    }

    SetVector<Value> live_ins;
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If the operand is defined in another block, it is a live-in value.
        if (auto block_arg = dyn_cast<BlockArgument>(operand)) {
          if (block_arg.getOwner() != &block) {
            live_ins.insert(operand);
          }
        } else {
          Operation *def_op = operand.getDefiningOp();
          if (def_op && def_op->getBlock() != &block) {
            live_ins.insert(operand);
          }
        }
      }
    }

    if (!live_ins.empty()) {
      direct_live_ins[&block] = live_ins;
    }
  }

  // If we update a branch or conditional branch, we may introduce new live-ins
  // for a block. So we need to propagate live-in values until a fixed point is
  // reached.

  // *************************************************************************
  // For example, consider this control flow:
  //
  // Block A:
  //   %0 = constant 1
  //   %1 = constant 2
  //   br B
  //
  // Block B:
  //   br C
  //
  // Block C:
  //   %2 = add %0, %1  // %0 and %1 are live-ins for block C
  //   return
  //
  // Initial direct_live_ins analysis:
  //   - Block C: {%0, %1} (directly used values defined outside C)
  //   - Block B: {} (no direct use of external values)
  //
  // After propagation:
  //   - Block C: {%0, %1}
  //   - Block B: {%0, %1} (needs to pass these values to C)
  //
  // The transformation adds block arguments:
  //   Block A:
  //     %0 = constant 1
  //     %1 = constant 2
  //     br B(%0, %1)
  //
  //   Block B(%b0, %b1):
  //     br C(%b0, %b1)
  //
  //   Block C(%c0, %c1):
  //     %2 = add %c0, %c1
  //     return
  // *************************************************************************
  DenseMap<Block *, SetVector<Value>> all_live_ins = direct_live_ins;
  bool changed = true;

  while (changed) {
    changed = false;

    for (Block &current_block : region.getBlocks()) {
      if (&current_block == &region.front()) {
        continue;
      }

      // Checks if current block has successor blocks and if they have
      // any live-ins.
      for (Block *succ_block : current_block.getSuccessors()) {
        auto succ_live_in_iter = all_live_ins.find(succ_block);
        if (succ_live_in_iter == all_live_ins.end()) {
          continue;
        }

        SetVector<Value> &succ_live_ins = succ_live_in_iter->second;
        SetVector<Value> &current_live_ins = all_live_ins[&current_block];

        // Checks if the live-in value in successor block is defined in the
        // current block.
        for (Value live_in : succ_live_ins) {
          // If it is defined in the current block, that means it is not a
          // live-in value for the current block. We can skip it.
          if (Operation *def_op = live_in.getDefiningOp()) {
            if (def_op->getBlock() == &current_block) {
              continue;
            }
          } else if (auto block_arg = dyn_cast<BlockArgument>(live_in)) {
            if (block_arg.getOwner() == &current_block) {
              continue;
            }
          }

          // If current live-ins do not contain the live-in value,
          // we add it to the current live-ins.
          if (!current_live_ins.contains(live_in)) {
            current_live_ins.insert(live_in);
            changed = true;
          }
        }
      }
    }
  }

  // llvm::errs() << "All live-ins after propagation:\n";
  // for (auto &[block, liveIns] : all_live_ins) {
  //   llvm::errs() << "Block: " << *block << "\nLive-ins: ";
  //   for (Value liveIn : liveIns) {
  //     llvm::errs() << "   " << liveIn << "\n";
  //   }
  //   llvm::errs() << "\n";
  // }

  // Adds all live-in values as block arguments and updates the
  // operations that use these live-in values.
  DenseMap<std::pair<Block *, Value>, Value> block_value_to_arg;
  DenseMap<Block *, unsigned> original_num_args;

  for (auto &[block, live_ins] : all_live_ins) {
    original_num_args[block] = block->getNumArguments();

    // Adds all live-in values as block arguments.
    for (Value value : live_ins) {
      block->addArgument(value.getType(), value.getLoc());
    }

    // Constructs a mapping from live-in values to their corresponding
    // block arguments.
    unsigned index = original_num_args[block];
    for (Value value : live_ins) {
      block_value_to_arg[{block, value}] = block->getArgument(index++);
    }
  }

  // Updates all operations in the region to use the new block arguments
  // instead of the live-in values.
  for (auto &[block, live_ins] : all_live_ins) {
    for (Operation &op : block->getOperations()) {
      for (unsigned i = 0; i < op.getNumOperands(); i++) {
        Value operand = op.getOperand(i);
        auto key = std::make_pair(block, operand);

        if (block_value_to_arg.count(key)) {
          op.setOperand(i, block_value_to_arg[key]);
        }
      }
    }
  }

  // Updates the terminators of predecessor blocks to use the new block
  // arguments instead of the live-in values.
  for (auto &[current_block, live_ins] : all_live_ins) {
    for (Block *pred_block : current_block->getPredecessors()) {
      Operation *term_op = pred_block->getTerminator();

      if (auto br_op = dyn_cast<neura::Br>(term_op)) {
        if (br_op.getDest() == current_block) {
          SmallVector<Value> new_operands(br_op.getOperands().begin(),
                                          br_op.getOperands().end());
          for (Value live_in : live_ins) {
            Operation *def_op = live_in.getDefiningOp();
            BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
            if (def_op && def_op->getBlock() == pred_block) {
              new_operands.push_back(live_in);
            } else if (block_arg && block_arg.getOwner() == pred_block) {
              new_operands.push_back(block_arg);
            } else if (all_live_ins[pred_block].contains(live_in)) {
              new_operands.push_back(block_value_to_arg[{pred_block, live_in}]);
            } else {
              assert(false && "Unexpected live-in value");
            }
          }
          OpBuilder builder(br_op);
          builder.create<neura::Br>(br_op.getLoc(), new_operands,
                                    current_block);
          br_op.erase();
        }
      } else if (auto cond_br_op = dyn_cast<neura::CondBr>(term_op)) {
        bool needs_update = false;
        SmallVector<Value> true_operands(cond_br_op.getTrueArgs().begin(),
                                         cond_br_op.getTrueArgs().end());
        SmallVector<Value> false_operands(cond_br_op.getFalseArgs().begin(),
                                          cond_br_op.getFalseArgs().end());
        // Handles the true branch.
        if (cond_br_op.getTrueDest() == current_block) {
          needs_update = true;
          for (Value live_in : live_ins) {
            Operation *def_op = live_in.getDefiningOp();
            BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
            if (def_op && def_op->getBlock() == pred_block) {
              true_operands.push_back(live_in);
            } else if (block_arg && block_arg.getOwner() == pred_block) {
              true_operands.push_back(block_arg);
            } else if (all_live_ins[pred_block].contains(live_in)) {
              true_operands.push_back(
                  block_value_to_arg[{pred_block, live_in}]);
            } else {
              assert(false && "Unexpected live-in value");
            }
          }
        }

        // Handles the false branch.
        if (cond_br_op.getFalseDest() == current_block) {
          needs_update = true;
          for (Value live_in : live_ins) {
            Operation *def_op = live_in.getDefiningOp();
            BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
            if (def_op && def_op->getBlock() == pred_block) {
              false_operands.push_back(live_in);
            } else if (block_arg && block_arg.getOwner() == pred_block) {
              false_operands.push_back(block_arg);
            } else if (all_live_ins[pred_block].contains(live_in)) {
              false_operands.push_back(
                  block_value_to_arg[{pred_block, live_in}]);
            } else {
              assert(false && "Unexpected live-in value");
            }
          }
        }

        // If an update is needed, create a new conditional branch operation.
        if (needs_update) {
          OpBuilder builder(cond_br_op);
          builder.create<neura::CondBr>(
              cond_br_op.getLoc(), cond_br_op.getCondition(), nullptr,
              true_operands, false_operands, cond_br_op.getTrueDest(),
              cond_br_op.getFalseDest());
          cond_br_op.erase();
        }
      }
    }
  }

  return success();
}

struct CanonicalizeLiveInPass
    : public PassWrapper<CanonicalizeLiveInPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeLiveInPass)

  StringRef getArgument() const override { return "canonicalize-live-in"; }
  StringRef getDescription() const override {
    return "Canonicalizes live-in values/operations in each basic block.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    module_op.walk([&](Operation *op) {
      Region *region = nullptr;
      if (auto func_op = dyn_cast<func::FuncOp>(op)) {
        auto accel_attr = func_op->getAttrOfType<StringAttr>("accelerator");
        if (!accel_attr || accel_attr.getValue() != "neura") {
          return;
        }
        region = &func_op.getBody();
      } else if (auto llvm_func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        auto accel_attr = llvm_func->getAttrOfType<StringAttr>("accelerator");
        if (!accel_attr || accel_attr.getValue() != "neura") {
          return;
        }
        region = &llvm_func.getBody();
      } else {
        return;
      }

      if (!region || region->empty()) {
        return;
      }

      if (failed(promoteFunctionArgsToConstants(*region))) {
        signalPassFailure();
        return;
      }

      if (failed(promoteLiveInValuesToBlockArgs(*region))) {
        signalPassFailure();
        return;
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createCanonicalizeLiveInPass() {
  return std::make_unique<CanonicalizeLiveInPass>();
}
} // namespace mlir::neura