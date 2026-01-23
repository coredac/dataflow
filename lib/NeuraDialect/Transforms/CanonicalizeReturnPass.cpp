#include "Common/AcceleratorAttrs.h"
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
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;

#define GEN_PASS_DEF_CANONICALIZERETURN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
// Return/Yield type attribute values.
constexpr const char *kReturnTypeAttr = "return_type";
constexpr const char *kYieldTypeAttr = "yield_type";
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

// Marks empty returns with "is_void" attribute and adds trigger values.
static void processReturns(Region &region, OpBuilder &builder) {
  SmallVector<neura::ReturnOp> empty_returns;

  region.walk([&](neura::ReturnOp ret_op) {
    llvm::errs() << "[ctrl2data] Processing neura.return operation...\n";
    llvm::errs() << ret_op << "\n";
    if (ret_op.getNumOperands() == 0) {
      empty_returns.push_back(ret_op);
    } else {
      llvm::errs() << "[ctrl2data] Marking neura.return with value...\n";
      ret_op->setAttr(kReturnTypeAttr, builder.getStringAttr(kReturnTypeValue));
    }
  });

  for (neura::ReturnOp ret_op : empty_returns) {
    ret_op->setAttr(kReturnTypeAttr, builder.getStringAttr(kReturnTypeVoid));
  }
}

// Processes neura.yield operations in kernel regions.
static void processYields(neura::KernelOp kernel_op, OpBuilder &builder) {
  SmallVector<neura::YieldOp> empty_yields;

  kernel_op.walk([&](neura::YieldOp yield_op) {
    llvm::errs() << "[canonicalize] Processing neura.yield operation...\n";
    llvm::errs() << yield_op << "\n";

    // Case 1: yield has results - mark as value type.
    if (yield_op.getResults().size() > 0) {
      llvm::errs() << "[canonicalize] Marking neura.yield with value...\n";
      yield_op->setAttr(kYieldTypeAttr,
                        builder.getStringAttr(kReturnTypeValue));
      return;
    }

    // Case 2 & 3: yield has no results.
    empty_yields.push_back(yield_op);
  });

  // Processes empty yields.
  for (neura::YieldOp yield_op : empty_yields) {
    llvm::errs() << "[canonicalize] Processing empty neura.yield...\n";

    // Searches for counters in the kernel.
    neura::CounterOp root_counter = nullptr;
    neura::CounterOp any_counter = nullptr;

    kernel_op.walk([&](neura::CounterOp counter_op) {
      any_counter = counter_op;

      if (counter_op.getCounterTypeAttr() &&
          counter_op.getCounterTypeAttr().getValue() == "root") {
        root_counter = counter_op;
      }
    });

    // Case 2: Has counter - uses counter as trigger.
    if (root_counter || any_counter) {
      Value trigger_value = root_counter ? root_counter.getCurrentIndex()
                                         : any_counter.getCurrentIndex();

      llvm::errs() << "[canonicalize] Using "
                   << (root_counter ? "root" : "leaf")
                   << " counter as trigger.\n";

      // Creates new yield with trigger value as result.
      builder.setInsertionPoint(yield_op);

      SmallVector<Value> iter_args_next(yield_op.getIterArgsNext());
      SmallVector<Value> results = {trigger_value};

      auto new_yield = builder.create<neura::YieldOp>(yield_op.getLoc(),
                                                      iter_args_next, results);
      new_yield->setAttr(kYieldTypeAttr,
                         builder.getStringAttr(kReturnTypeVoid));

      yield_op.erase();
    } else {
      // Case 3: No counter - mark for void processing (similar to return).
      llvm::errs()
          << "[canonicalize] No counter found, marking as void yield\n";
      yield_op->setAttr(kYieldTypeAttr, builder.getStringAttr(kReturnTypeVoid));
    }
  }
}

// Processes empty yield void blocks (similar to processEmptyReturnVoidBlock).
static void processEmptyYieldVoidBlock(Block *yield_block,
                                       neura::YieldOp void_yield_op,
                                       OpBuilder &builder) {
  SmallVector<Block *> predecessor_blocks(yield_block->getPredecessors());

  // Entry block with yield_void is unreachable; no action needed.
  if (predecessor_blocks.empty()) {
    llvm::errs()
        << "[canonicalize] Entry block with void yield is unreachable\n";
    return;
  }

  // Separates predecessor blocks into cond_br and br blocks.
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

  // Handles br_preds: copy yield_void to pred_block with a trigger value.
  for (Block *pred_block : br_preds) {
    neura::Br br = cast<neura::Br>(pred_block->getTerminator());

    // Finds a suitable trigger value in the predecessor block.
    Value trigger_value = nullptr;

    for (Operation &op : llvm::reverse(*pred_block)) {
      if (&op == br) {
        continue;
      }

      if (op.getNumResults() > 0) {
        trigger_value = op.getResult(0);
        break;
      }
    }

    if (!trigger_value) {
      llvm::errs() << "[canonicalize] Error: No suitable value found in "
                      "predecessor block\n";
      return;
    }

    builder.setInsertionPoint(br);

    SmallVector<Value> iter_args_next(void_yield_op.getIterArgsNext());
    SmallVector<Value> results = {trigger_value};

    auto new_yield =
        builder.create<neura::YieldOp>(br.getLoc(), iter_args_next, results);
    new_yield->setAttr(kYieldTypeAttr, builder.getStringAttr(kReturnTypeVoid));
    br.erase();
  }

  // If there are no cond_br predecessors, remove the yield_void block.
  if (cond_br_preds.empty()) {
    void_yield_op.erase();
    yield_block->erase();
    return;
  }

  // Handles cond_preds: add a block argument for the trigger value.
  BlockArgument trigger_arg =
      yield_block->addArgument(builder.getI1Type(), void_yield_op.getLoc());

  // Updates each cond_pred block's terminator to pass the trigger value.
  for (Block *pred_block : cond_br_preds) {
    neura::CondBr cond_br = cast<neura::CondBr>(pred_block->getTerminator());
    Value cond = cond_br.getCondition();
    Value trigger_value = nullptr;

    bool is_true_branch = (cond_br.getTrueDest() == yield_block);
    bool is_false_branch = (cond_br.getFalseDest() == yield_block);

    if (is_true_branch && !is_false_branch) {
      trigger_value = cond;
    } else if (!is_true_branch && is_false_branch) {
      builder.setInsertionPoint(cond_br);
      Value negated_cond =
          builder.create<neura::NotOp>(cond_br.getLoc(), cond.getType(), cond);
      trigger_value = negated_cond;
    } else {
      llvm::errs() << "[canonicalize] Error: Both branches lead to yield\n";
      return;
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

  // Updates the yield_void operation to use the block argument as trigger.
  builder.setInsertionPoint(void_yield_op);

  SmallVector<Value> iter_args_next(void_yield_op.getIterArgsNext());
  SmallVector<Value> results = {trigger_arg};

  auto new_yield = builder.create<neura::YieldOp>(void_yield_op.getLoc(),
                                                  iter_args_next, results);
  new_yield->setAttr(kYieldTypeAttr, builder.getStringAttr(kReturnTypeVoid));
  void_yield_op.erase();
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
    // In dataflow semantics, even void returns need a value to trigger them.
    // We use the result of the last operation in the predecessor block as the
    // trigger signal.
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
    : public PassWrapper<CanonicalizeReturnPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReturnPass)

  StringRef getArgument() const override { return "canonicalize-return"; }
  StringRef getDescription() const override {
    return "Canonicalizes return operations in Neura functions.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<neura::NeuraDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    OpBuilder builder(module_op.getContext());

    // Processes all functions.
    module_op.walk([&](func::FuncOp func_op) {
      // Checks for neura accelerator attribute.
      auto accel_attr =
          func_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr) {
        return;
      }

      Region &region = func_op.getBody();
      if (region.empty()) {
        return;
      }

      // Step 1: Marks empty returns with "void" attribute.
      processReturns(region, builder);

      if (!isVoidFunction(func_op)) {
        llvm::errs() << "[ctrl2data] Function is not void, no further action "
                        "needed.\n";
        return;
      }

      // Step 2: Collects all return operations with "is_void" attribute.
      SmallVector<neura::ReturnOp> ret_void_ops;
      region.walk([&](neura::ReturnOp ret_op) {
        if (ret_op->hasAttr(kReturnTypeAttr)) {
          if (dyn_cast<StringAttr>(ret_op->getAttr(kReturnTypeAttr))
                  .getValue() == kReturnTypeVoid) {
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
          // return_void operation, and redirect the original return block to
          // this new block.
          assert(false && "Unsupported case: return block is not empty.");
        }
      }
    });

    // Processes all neura.kernel operations.
    module_op.walk([&](neura::KernelOp kernel_op) {
      auto accel_attr =
          kernel_op->getAttrOfType<StringAttr>(accel::kAcceleratorAttr);
      if (!accel_attr) {
        return;
      }

      // Step 1: Processes yields (handles cases 1 & 2)
      processYields(kernel_op, builder);

      // Step 2: Collects void yields without trigger values (case 3).
      SmallVector<neura::YieldOp> yield_void_ops;
      kernel_op.walk([&](neura::YieldOp yield_op) {
        if (yield_op->hasAttr(kYieldTypeAttr)) {
          if (dyn_cast<StringAttr>(yield_op->getAttr(kYieldTypeAttr))
                      .getValue() == kReturnTypeVoid &&
              yield_op.getResults().size() == 0) {
            yield_void_ops.push_back(yield_op);
          }
        }
      });

      // Step 3: Processes each yield_void block (case 3)
      for (neura::YieldOp yield_void_op : yield_void_ops) {
        Block *yield_block = yield_void_op->getBlock();
        bool is_empty_block = (yield_block->getOperations().size() == 1);

        if (is_empty_block) {
          processEmptyYieldVoidBlock(yield_block, yield_void_op, builder);
        } else {
          assert(false && "Unsupported case: yield block is not empty.");
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::neura::createCanonicalizeReturnPass() {
  return std::make_unique<CanonicalizeReturnPass>();
}