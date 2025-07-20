#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

#define GEN_PASS_DEF_NEURACANONICALIZE
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
LogicalResult promoteLiveInValuesToBlockArgs(Operation *op) {
  auto func_op = dyn_cast<func::FuncOp>(op);
  if (!func_op)
    return success();

  for (Block &block : func_op.getBlocks()) {
    // Skips the entry block.
    if (&block == &func_op.getBlocks().front())
      continue;

    // Identifies all the live-in values in the block.
    llvm::SetVector<Value> live_ins;

    // Iterates over each operation in the block and its operands.
    for (Operation &op : block.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If the operand is not a block argument and is defined outside the
        // current block, it is a live-in value.
        if (!dyn_cast<BlockArgument>(operand)) {
          Operation *defOp = operand.getDefiningOp();
          if (defOp && defOp->getBlock() != &block) {
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
    unsigned originalNumArgs = block.getNumArguments();
    for (Value value : live_ins) {
      block.addArgument(value.getType(), value.getLoc());
    }

    // Creates a mapping from live-in values to the new block arguments.
    DenseMap<Value, Value> value_to_arg;
    for (unsigned i = 0; i < live_ins.size(); ++i) {
      value_to_arg[live_ins[i]] = block.getArgument(originalNumArgs + i);
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

          // Adds live-in values as new operands
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

struct NeuraCanonicalizePass
    : public PassWrapper<NeuraCanonicalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NeuraCanonicalizePass)

  StringRef getArgument() const override { return "neura-canonicalize"; }
  StringRef getDescription() const override {
    return "Canonicalizes operations in the Neura dialect";
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    module_op->walk([&](func::FuncOp func) {
      if (failed(promoteLiveInValuesToBlockArgs(func))) {
        signalPassFailure();
        return;
      }
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createNeuraCanonicalizePass() {
  return std::make_unique<NeuraCanonicalizePass>();
}
} // namespace mlir::neura