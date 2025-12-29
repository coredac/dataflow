#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
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
// Return type attribute values.
constexpr const char *kReturnTypeAttr = "return_type";
constexpr const char *kReturnTypeVoid = "void";
constexpr const char *kReturnTypeValue = "value";

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

// Marks empty returns with "is_void" attribute and adds trigger values.
static void processVoidReturns(Region &region, OpBuilder &builder) {
  SmallVector<neura::ReturnOp> empty_returns;

  region.walk([&](neura::ReturnOp ret_op) {
    if (ret_op.getNumOperands() == 0) {
      empty_returns.push_back(ret_op);
    } else {
      ret_op->setAttr(kReturnTypeAttr, builder.getStringAttr(kReturnTypeValue));
    }
  });

  for (neura::ReturnOp ret_op : empty_returns) {
    ret_op->setAttr(kReturnTypeAttr, builder.getStringAttr(kReturnTypeVoid));
  }
}

static void processEmptyReturnVoidBlock(Block *ret_block,
                                        neura::ReturnOp void_ret_op,
                                        OpBuilder &builder) {
  SmallVector<Block *> predecessor_blocks(ret_block->getPredecessors());
  // Entry bolock with return_void is unreachable; no action needed.
  if (predecessor_blocks.empty()) {
    assert(false && "Entry block with neura.return_void is unreachable.");
  }

  // Seperates predecessor blocks into cond_br and br blocks.
  SmallVector<Block *> cond_br_preds;
  SmallVector<Block *> br_preds;

  for (Block *pred_block : predecessor_blocks) {
    Operation *terminator = pred_block->getTerminator();
    if (isa<neura::CondBr>(terminator)) {
      cond_br_preds.push_back(pred_block);
    } else if (isa<neura::Br>(terminator)) {
      br_preds.push_back(pred_block);
    }
  }

  // Handles br_preds: copies return_void to pred_block, and utilizes a suitable
  // value to trigger it.
  for (Block *pred_block : br_preds) {
    neura::Br br = cast<neura::Br>(pred_block->getTerminator());

    // Finds a suitable trigger value in the predecessor block.
    Value trigger_value = nullptr;

    // Iterates through operations in reverse order to find the last suitable
    // value.
    for (Operation &op : llvm::reverse(*pred_block)) {
      // Skips the terminator itself.
      if (&op == br) {
        continue;
      }

      // Looks for any suitable value in the predecessor block.
      if (op.getNumResults() > 0) {
        trigger_value = op.getResult(0);
        break;
      }
    }

    if (!trigger_value) {
      assert(false && "No suitable value found in predecessor block.");
    }

    builder.setInsertionPoint(br);
    auto new_ret = builder.create<neura::ReturnOp>(br.getLoc(), trigger_value);
    new_ret->setAttr(kReturnTypeAttr, builder.getStringAttr(kReturnTypeVoid));
    br.erase();
  }

  // If there are no cond_br predecessors, removes the return_void block.
  if (cond_br_preds.empty()) {
    void_ret_op.erase();
    ret_block->erase();
    return;
  }

  // Handles cond_preds: adds a block argument for the trigger value, and
  // updates each predecessor's terminator to pass the trigger value.
  BlockArgument trigger_arg =
      ret_block->addArgument(builder.getI1Type(), void_ret_op.getLoc());

  // Updates each cond_pred block's terminator to pass the trigger value.
  for (Block *pred_block : cond_br_preds) {
    neura::CondBr cond_br = cast<neura::CondBr>(pred_block->getTerminator());
    Value cond = cond_br.getCondition();
    Value trigger_value = nullptr;

    bool is_true_branch = (cond_br.getTrueDest() == ret_block);
    bool is_false_branch = (cond_br.getFalseDest() == ret_block);

    if (is_true_branch && !is_false_branch) {
      // True branck leads to return_void, uses condition directly.
      trigger_value = cond;
    } else if (!is_true_branch && is_false_branch) {
      // False branch leads to return_void, uses negated condition.
      builder.setInsertionPoint(cond_br);
      Value negated_cond =
          builder.create<neura::NotOp>(cond_br.getLoc(), cond.getType(), cond);
      trigger_value = negated_cond;
    } else {
      assert(false && "Unsupported case: Both branches lead to neura.return.");
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
  }
  // Updates the return_void operation to use the block argument as trigger.
  builder.setInsertionPoint(void_ret_op);
  auto new_ret =
      builder.create<neura::ReturnOp>(void_ret_op.getLoc(), trigger_arg);
  new_ret->setAttr(kReturnTypeAttr, builder.getStringAttr(kReturnTypeVoid));
  void_ret_op.erase();
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

    // Step 1: Marks empty returns with "is_void" attribute.
    processVoidReturns(region, builder);

    // Step 2: Collects all return operations with "is_void" attribute.
    SmallVector<neura::ReturnOp> ret_void_ops;
    region.walk([&](neura::ReturnOp ret_op) {
      if (ret_op->hasAttr(kReturnTypeAttr)) {
        if (dyn_cast<StringAttr>(ret_op->getAttr(kReturnTypeAttr)).getValue() ==
            kReturnTypeVoid) {
          ret_void_ops.push_back(ret_op);
        }
      }
    });

    // Step 3: Processes each return_void block.
    for (neura::ReturnOp ret_void_op : ret_void_ops) {
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