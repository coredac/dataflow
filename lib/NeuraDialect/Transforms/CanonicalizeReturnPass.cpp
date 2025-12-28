#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZERETURN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// Checks if a function is a void function (i.e., has no return values).
static bool isVoidFunction(func::FuncOp func_op) {
  if (func_op.getNumResults() == 0) {
    return true;
  }
  if (func_op.getNumResults() == 1) {
    if (isa<LLVM::LLVMVoidType>(func_op.getResultTypes()[0])) {
      return true;
    }
  }
  return false;
}

// Converts neura.return with no operands to neura.return_void.
static void convertEmptyReturnToReturnVoid(Region &region, OpBuilder &builder) {
  SmallVector<neura::ReturnOp> empty_returns;

  region.walk([&](neura::ReturnOp ret_op) {
    if (ret_op.getNumOperands() == 0) {
      empty_returns.push_back(ret_op);
    }
  });

  for (neura::ReturnOp ret_op : empty_returns) {
    builder.setInsertionPoint(ret_op);
    builder.create<neura::ReturnVoidOp>(ret_op.getLoc(), Value{});
    ret_op.erase();
  }
}

static void processEmptyReturnVoidBlock(Block *ret_block,
                                        neura::ReturnVoidOp ret_void_op,
                                        OpBuilder &builder) {
  SmallVector<Block *> predecessor_blocks(ret_block->getPredecessors());
  // Entry bolock with return_void is unreachable; no action needed.
  if (predecessor_blocks.empty()) {
    assert(false && "Entry block with neura.return_void is unreachable.");
  }

  // Adds a block argument for the trigger value.
  BlockArgument trigger_arg =
      ret_block->addArgument(builder.getI1Type(), ret_void_op.getLoc());

  // Updates each predecessor's terminator to pass the trigger value.
  for (Block *pred_block : predecessor_blocks) {
    Operation *terminator = pred_block->getTerminator();
    if (auto cond_br = dyn_cast<neura::CondBr>(terminator)) {
      Value cond = cond_br.getCondition();
      Value trigger_value = nullptr;

      bool is_true_branch = (cond_br.getTrueDest() == ret_block);
      bool is_false_branch = (cond_br.getFalseDest() == ret_block);

      if (is_true_branch && !is_false_branch) {
        // True branck leads to return_void, uses condition directly.
        trigger_value = cond;
      } else if (!is_true_branch && is_false_branch) {
        // False branch leads to return_void, uses negated condition.
        builder.setInsertionPoint(terminator);
        Value negated_cond = builder.create<neura::NotOp>(terminator->getLoc(),
                                                          cond.getType(), cond);
        trigger_value = negated_cond;
      } else {
        assert(false &&
               "Unsupported case: Both branches lead to neura.return_void.");
      }

      if (trigger_value) {
        SmallVector<Value> true_args(cond_br.getTrueArgs());
        SmallVector<Value> false_args(cond_br.getFalseArgs());

        if (is_true_branch) {
          true_args.push_back(trigger_value);
        }
        if (is_false_branch) {
          false_args.push_back(trigger_value);
        }

        builder.setInsertionPoint(cond_br);
        builder.create<neura::CondBr>(
            cond_br.getLoc(), cond_br.getCondition(), true_args, false_args,
            cond_br.getTrueDest(), cond_br.getFalseDest());
        cond_br.erase();
      }
    } else if (auto br = dyn_cast<neura::Br>(terminator)) {
      Value trigger_value;

      // Looks for any suitable value in the predecessor block to use as
      // trigger.
      for (Operation &op : llvm::reverse(*pred_block)) {
        if (&op == terminator) {
          continue;
        }
        if (op.getNumResults() > 0) {
          trigger_value = op.getResult(0);
          break;
        }
      }

      // If no suitable value is found, reports an error.
      if (!trigger_value) {
        assert(false && "No suitable value found for trigger.");
      }

      SmallVector<Value> args(br.getArgs());
      args.push_back(trigger_value);

      builder.setInsertionPoint(br);
      builder.create<neura::Br>(br.getLoc(), args, br.getDest());
      br.erase();
    }
  }
  // Updates the return_void operation to use the block argument as trigger.
  builder.setInsertionPoint(ret_void_op);
  builder.create<neura::ReturnVoidOp>(ret_void_op.getLoc(), trigger_arg);
  ret_void_op.erase();
}

struct CanonicalizeReturnPass
    : public PassWrapper<CanonicalizeReturnPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReturnPass)

  StringRef getArgument() const override { return "canonicalize-return"; }
  StringRef getDescription() const override {
    return "Canonicalizes return operations in Neura functions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    // Checks for neura accelerator attribute.
    auto accel_attr = func_op->getAttrOfType<StringAttr>("accelerator");
    if (!accel_attr) {
      return;
    }

    // Skips non-void functions.
    if (!isVoidFunction(func_op)) {
      return;
    }

    Region &region = func_op.getBody();
    if (region.empty()) {
      return;
    }

    OpBuilder builder(func_op.getContext());

    // Step 1: Converts empty neura.return to neura.return_void.
    convertEmptyReturnToReturnVoid(region, builder);

    // Step 2: Collects all return_void operations without triggers.
    SmallVector<neura::ReturnVoidOp> ret_void_ops;
    region.walk([&](neura::ReturnVoidOp ret_void_op) {
      if (!ret_void_op.getTrigger()) {
        ret_void_ops.push_back(ret_void_op);
      }
    });

    // Step 3: Processes each return_void block.
    for (neura::ReturnVoidOp ret_void_op : ret_void_ops) {
      Block *ret_block = ret_void_op->getBlock();

      // Checks if ret_block only contains the return_void operation.
      bool is_empty_block = (ret_block->getOperations().size() == 1);

      if (is_empty_block) {
        processEmptyReturnVoidBlock(ret_block, ret_void_op, builder);
      } else {
        // TODO: Handle non-empty return blocks.
        // The basic idea is to create a new block that only contains the
        // return_void operation, and redirect the original return block to this
        // new block.
        assert(false && "Unsupported case: return block is not empty.");
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createCanonicalizeReturnPass() {
  return std::make_unique<CanonicalizeReturnPass>();
}