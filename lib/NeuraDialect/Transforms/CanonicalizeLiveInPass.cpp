#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZELIVEIN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct DirectDataflowLiveIn {
  // The live-in value.
  Value value;
  // The block where the live-in value is defined.
  Block *defining_block;
  // The block where the live-in value is used.
  Block *using_block;
};

DenseMap<Block *, SmallVector<DirectDataflowLiveIn>>
identifyDirectDataflowLiveIns(Region &region, DominanceInfo &dom_info,
                              PostDominanceInfo &post_dom_info) {
  DenseMap<Block *, SmallVector<DirectDataflowLiveIn>>
      using_block_to_direct_dataflow_live_ins;
  for (Block &block : region.getBlocks()) {
    llvm::errs() << "[CanoLiveIn] Analyzing block:\n" << block << "\n";
    // Skips the entry block.
    if (&block == &region.front()) {
      continue;
    }

    // Collects direct live-in values for the block.
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

    // Checks each live-in value to see if it has direct dataflow dependency.
    for (Value live_in : live_ins) {
      llvm::errs() << "[CanoLiveIn] Checking live-in value: " << live_in
                   << "\n";
      Block *defining_block = nullptr;
      bool is_block_arg = false;

      if (auto block_arg = dyn_cast<BlockArgument>(live_in)) {
        defining_block = block_arg.getOwner();
        is_block_arg = true;
      } else {
        Operation *def_op = live_in.getDefiningOp();
        if (def_op) {
          defining_block = def_op->getBlock();
        }
      }

      if (!defining_block) {
        llvm::errs()
            << "[CanoLiveIn] Error: Unable to find defining block for live-in "
               "value: "
            << live_in << "\n";
        continue;
      }

      // Step 1: BB_a dominates BB_b.
      if (!dom_info.dominates(defining_block, &block)) {
        llvm::errs() << "[CanoLiveIn] Info: Defining block does not dominate "
                        "using block for live-in "
                        "value: "
                     << live_in << "\n";
        continue;
      }

      // Step 2: BB_b post-dominates BB_a.
      if (!post_dom_info.postDominates(&block, defining_block)) {
        llvm::errs() << "[CanoLiveIn] Info: Using block does not post-dominate "
                        "defining block for live-in "
                        "value: "
                     << live_in << "\n";
        continue;
      }

      // Step 3: Checks if the live-in value is a block argument defined in the
      // defining block.
      if (is_block_arg) {
        llvm::errs() << "[CanoLiveIn] Info: Live-in value is a block argument: "
                     << live_in << "\n";
        // continue;
      }

      DirectDataflowLiveIn direct_dataflow_live_in;
      direct_dataflow_live_in.value = live_in;
      direct_dataflow_live_in.defining_block = defining_block;
      direct_dataflow_live_in.using_block = &block;

      using_block_to_direct_dataflow_live_ins[&block].push_back(
          direct_dataflow_live_in);

      llvm::errs() << "[CanoLiveIn] Found direct dataflow live-in value: \n";
      llvm::errs() << "  Value: " << live_in << "\n";
      llvm::errs() << "  Defining Block: \n" << *defining_block << "\n";
      llvm::errs() << "  Using Block: \n" << block << "\n";
    }
  }
  return using_block_to_direct_dataflow_live_ins;
}

LogicalResult promoteLiveInValuesToBlockArgs(Region &region,
                                             DominanceInfo &dom_info,
                                             PostDominanceInfo &post_dom_info) {
  if (region.empty()) {
    return success();
  }

  DenseMap<Block *, SmallVector<DirectDataflowLiveIn>>
      direct_dataflow_live_ins =
          identifyDirectDataflowLiveIns(region, dom_info, post_dom_info);

  // Maps each block to its direct dataflow live-in values.
  DenseMap<Block *, SetVector<Value>> direct_dataflow_live_in_values;
  for (auto &[block, dataflow_live_ins] : direct_dataflow_live_ins) {
    for (auto &dataflow_live_in : dataflow_live_ins) {
      direct_dataflow_live_in_values[block].insert(dataflow_live_in.value);
    }
  }

  // Collects direct live-in values for each block in the region.
  // Without considering the transitive dependencies.
  DenseMap<Block *, SetVector<Value>> direct_live_ins;

  // Initializes the direct live-ins for each block.
  for (Block &block : region.getBlocks()) {
    if (&block == &region.front()) {
      continue;
    }

    SetVector<Value> live_ins;
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If the operand is a direct dataflow live-in value, skip it.
        if (direct_dataflow_live_in_values[&block].contains(operand)) {
          continue;
        }

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

  // If we update a branch or conditional branch, we may introduce new
  // live-ins for a block. So we need to propagate live-in values until a
  // fixed point is reached.

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
          // If it is a direct dataflow live-in value for the successor block,
          // we skip it.
          if (direct_dataflow_live_in_values[succ_block].contains(live_in)) {
            continue;
          }

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

  // Adds block arguments for direct dataflow live-in values.
  // for (auto &[block, dataflow_live_ins] : direct_dataflow_live_ins) {
  //   if (original_num_args.find(block) == original_num_args.end()) {
  //     original_num_args[block] = block->getNumArguments();
  //   }

  //   for (auto &dataflow_live_in : dataflow_live_ins) {
  //     block->addArgument(dataflow_live_in.value.getType(),
  //                        dataflow_live_in.value.getLoc());
  //     unsigned index = block->getNumArguments() - 1;
  //     block_value_to_arg[{block, dataflow_live_in.value}] =
  //         block->getArgument(index);
  //     llvm::errs()
  //         << "[CanoLiveIn] Added block argument for direct dataflow live-in "
  //            "value: \n";
  //     llvm::errs() << "  Value: " << dataflow_live_in.value << "\n";
  //     llvm::errs() << "  Defining Block: \n"
  //                  << *dataflow_live_in.defining_block << "\n";
  //     llvm::errs() << "  Using Block: \n" << *block << "\n";
  //   }
  // }

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
              cond_br_op.getLoc(), cond_br_op.getCondition(), true_operands,
              false_operands, cond_br_op.getTrueDest(),
              cond_br_op.getFalseDest());
          cond_br_op.erase();
        }
      }
    }
  }

  // for (auto &[use_block, directFlows] : direct_dataflow_live_ins) {
  //   for (auto &flow : directFlows) {
  //     Block *def_block = flow.defining_block;
  //     Value value = flow.value;

  //     // 检查def_block是否是use_block的直接前驱
  //     bool is_direct_pred = false;
  //     for (Block *pred : use_block->getPredecessors()) {
  //       if (pred == def_block) {
  //         is_direct_pred = true;
  //         break;
  //       }
  //     }

  //     if (is_direct_pred) {
  //       // 情况1：def_block直接跳转到use_block
  //       // 需要更新def_block的终止符，添加value作为操作数
  //       Operation *term_op = def_block->getTerminator();

  //       if (auto br_op = dyn_cast<neura::Br>(term_op)) {
  //         if (br_op.getDest() == use_block) {
  //           SmallVector<Value> new_operands(br_op.getOperands().begin(),
  //                                           br_op.getOperands().end());
  //           new_operands.push_back(value);

  //           OpBuilder builder(br_op);
  //           builder.create<neura::Br>(br_op.getLoc(), new_operands,
  //           use_block); br_op.erase();

  //           llvm::errs() << "更新直接前驱的Br终止符，添加直接数据流操作数\n";
  //         }
  //       } else if (auto cond_br_op = dyn_cast<neura::CondBr>(term_op)) {
  //         SmallVector<Value> true_operands(cond_br_op.getTrueArgs().begin(),
  //                                          cond_br_op.getTrueArgs().end());
  //         SmallVector<Value>
  //         false_operands(cond_br_op.getFalseArgs().begin(),
  //                                           cond_br_op.getFalseArgs().end());

  //         if (cond_br_op.getTrueDest() == use_block) {
  //           true_operands.push_back(value);
  //         }
  //         if (cond_br_op.getFalseDest() == use_block) {
  //           false_operands.push_back(value);
  //         }

  //         OpBuilder builder(cond_br_op);
  //         builder.create<neura::CondBr>(
  //             cond_br_op.getLoc(), cond_br_op.getCondition(), true_operands,
  //             false_operands, cond_br_op.getTrueDest(),
  //             cond_br_op.getFalseDest());
  //         cond_br_op.erase();

  //         llvm::errs() <<
  //         "更新直接前驱的CondBr终止符，添加直接数据流操作数\n";
  //       }
  //     } else {
  //       // 情况2：def_block不是use_block的直接前驱
  //       // 需要找到从def_block到use_block路径上的所有中间块，并传递value
  //       // 这个情况比较复杂，但在满足后支配条件下，通常不会发生
  //       llvm::errs() <<
  //       "警告：直接数据流live-in的定义块不是使用块的直接前驱\n";
  //     }
  //   }
  // }

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

      DominanceInfo dom_info(op);
      PostDominanceInfo post_dom_info(op);

      if (failed(promoteLiveInValuesToBlockArgs(*region, dom_info,
                                                post_dom_info))) {
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