#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Casting.h"
#include <string>

using namespace mlir;

#define GEN_PASS_DEF_NEURACANONICALIZE
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// Returns blocks in a region in topological order
SmallVector<Block *> getBlocksInTopologicalOrder(Region &region) {
  if (region.empty())
    return {};

  SmallVector<Block *> ordered_blocks;

  DenseSet<Block *> visited;

  std::function<void(Block *)> dfs = [&](Block *block) {
    visited.insert(block);

    Operation *terminator = block->getTerminator();
    if (terminator) {
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        Block *succ = terminator->getSuccessor(i);
        if (!visited.count(succ)) {
          dfs(succ);
        }
      }
    }
    ordered_blocks.push_back(block);
  };

  dfs(&region.front());

  for (Block &block : region) {
    if (!visited.count(&block)) {
      dfs(&block);
    }
  }

  std::reverse(ordered_blocks.begin(), ordered_blocks.end());
  return ordered_blocks;
}

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

  SmallVector<Block *> sorted_blocks = getBlocksInTopologicalOrder(region);

  for (Block *block_ptr : sorted_blocks) {
    Block &block = *block_ptr;
    // Skips the entry block.
    if (&block == &region.front())
      continue;

    // Identifies all the live-in values in the block.
    llvm::SetVector<Value> live_ins;

    // Iterates over each operation in the block and its operands.
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If the operand is not a block argument and is defined outside the
        // current block, it is a live-in value.
        if (!dyn_cast<BlockArgument>(operand)) {
          Operation *def_op = operand.getDefiningOp();
          if (def_op && def_op->getBlock() != &block) {
            live_ins.insert(operand);
          }
        } else if (dyn_cast<BlockArgument>(operand).getOwner() != &block) {
          // If it is a block argument but defined in another block,
          // it is also considered a live-in value.
          live_ins.insert(operand);
        }
      }
    }

    if (live_ins.empty())
      continue;

    // Adds new block arguments for each live-in value.
    unsigned original_num_args = block.getNumArguments();
    for (Value value : live_ins) {
      block.addArgument(value.getType(), value.getLoc());
    }

    // Creates a mapping from live-in values to the new block arguments.
    DenseMap<Value, Value> value_to_arg;
    for (unsigned i = 0; i < live_ins.size(); ++i) {
      value_to_arg[live_ins[i]] = block.getArgument(original_num_args + i);
    }

    // Updates all operations in the block to use the new block arguments
    // instead of the live-in values.
    for (Operation &op : block.getOperations()) {
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value operand = op.getOperand(i);
        auto it = value_to_arg.find(operand);
        if (it != value_to_arg.end()) {
          op.setOperand(i, it->second);
        }
      }
    }

    // Updates the terminator of predecessor blocks to include the new block
    // arguments.
    for (Block *pred_block : block.getPredecessors()) {
      Operation *pred_op = pred_block->getTerminator();
      // Handles br operations.
      if (auto br_op = dyn_cast<neura::Br>(pred_op)) {
        if (br_op.getDest() == &block) {
          // Creates a new operand list, including the original operands.
          SmallVector<Value, 4> new_operands;

          for (Value operand : br_op.getOperands()) {
            new_operands.push_back(operand);
          }

          // Adds live-in values as new operands.
          for (Value live_in : live_ins) {
            new_operands.push_back(live_in);
          }

          // Creates a new branch operation with the updated operands.
          OpBuilder builder(br_op);
          builder.create<neura::Br>(br_op.getLoc(), new_operands, &block);

          // Erases the old branch operation.
          br_op.erase();
        }
      }
      // Handles conditional branch operations.
      else if (auto cond_br_op = dyn_cast<neura::CondBr>(pred_op)) {
        OpBuilder builder(cond_br_op);
        bool needs_update = false;

        SmallVector<Value, 4> true_operands, false_operands;
        Block *true_dest = cond_br_op.getTrueDest();
        Block *false_dest = cond_br_op.getFalseDest();

        for (Value operand : cond_br_op.getTrueArgs()) {
          true_operands.push_back(operand);
        }
        for (Value operand : cond_br_op.getFalseArgs()) {
          false_operands.push_back(operand);
        }

        // Checks if the true branch destination is the current block.
        if (true_dest == &block) {
          needs_update = true;
          for (Value live_in : live_ins) {
            true_operands.push_back(live_in);
          }
        }

        // Checks if the false branch destination is the current block.
        if (false_dest == &block) {
          needs_update = true;
          for (Value live_in : live_ins) {
            false_operands.push_back(live_in);
          }
        }

        if (needs_update) {
          // Predicated bit defaults to null.
          builder.create<neura::CondBr>(
              cond_br_op.getLoc(), cond_br_op.getCondition(), nullptr,
              true_operands, false_operands, true_dest, false_dest);

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