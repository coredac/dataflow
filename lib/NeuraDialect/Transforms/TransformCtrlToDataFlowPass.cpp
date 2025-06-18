#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

#define GEN_PASS_DEF_TransformCtrlToDataFlow
#include "NeuraDialect/NeuraPasses.h.inc"

// Inserts `grant_once` for every predicated value defined in the entry block
// that is used outside of the block (i.e., a live-out).
void insertGrantOnceInEntryBlock(Block *entry_block, OpBuilder &builder,
                                 DenseMap<Value, Value> &granted_once_map) {
  SmallVector<Value> live_out_values;

  // Step 1: Collects all live-out values first.
  for (Operation &op : *entry_block) {
    for (Value result : op.getResults()) {
      if (!isa<neura::PredicatedValue>(result.getType()))
        continue;

      bool is_live_out = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return user->getBlock() != entry_block || isa<neura::Br, neura::CondBr>(user);
      });

      if (is_live_out && !granted_once_map.contains(result))
        live_out_values.push_back(result);
    }
  }

  // Step 2: Inserts grant_once for each candidate.
  for (Value val : live_out_values) {
    Operation *def_op = val.getDefiningOp();
    if (!def_op)
      continue;

    builder.setInsertionPointAfter(def_op);
    auto granted = builder.create<neura::GrantOnceOp>(def_op->getLoc(), val.getType(), val);
    granted_once_map[val] = granted.getResult();

    // Replaces external uses with granted result.
    for (OpOperand &use : llvm::make_early_inc_range(val.getUses())) {
      Operation *user = use.getOwner();
      if (user->getBlock() != entry_block || isa<neura::Br, neura::CondBr>(user)) {
        use.set(granted.getResult());
      }
    }
  }
}

// Returns blocks in post-order traversal order.
void getBlocksInPostOrder(Block *startBlock, SmallVectorImpl<Block *> &postOrder,
                         DenseSet<Block *> &visited) {
  if (!visited.insert(startBlock).second)
    return;

  // Visits successors first.
  for (Block *succ : startBlock->getSuccessors())
    getBlocksInPostOrder(succ, postOrder, visited);

  // Adds current block to post-order sequence.
  postOrder.push_back(startBlock);
}

// Creates phi nodes for all live-in values in the given block.
void createPhiNodesForBlock(
    Block *block, OpBuilder &builder,
    SmallVectorImpl<std::tuple<Value, Value, Value, Block *>> &deferred_ctrl_movs) {
  if (block->hasNoPredecessors()) {
    // Skips phi insertion for entry block.
    return;
  }

  // Collects all live-in values.
  std::vector<Value> live_ins;
  for (Operation &op : *block) {
    for (Value operand : op.getOperands()) {
      // Identifies operands defined in other blocks.
      if (operand.getDefiningOp() &&
          operand.getDefiningOp()->getBlock() != block) {
        live_ins.push_back(operand);
        continue;
      }
      // Collects all block arguments.
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(operand)) {
        live_ins.push_back(operand);
      }
    }
  }

  builder.setInsertionPointToStart(block);
  for (Value live_in : live_ins) {
    // Creates predicated type for phi node.
    Type live_in_type = live_in.getType();
    Type predicated_type = isa<neura::PredicatedValue>(live_in_type)
        ? live_in_type
        : neura::PredicatedValue::get(builder.getContext(), live_in_type, builder.getI1Type());

    // Uses the location from the first operation in the block or block's parent operation.
    Location loc = block->empty() ?
                   block->getParent()->getLoc() :
                   block->front().getLoc();
    SmallVector<Value> phi_operands;
    llvm::SmallDenseSet<Operation*, 4> just_created_consumer_ops;
    BlockArgument arg = dyn_cast<BlockArgument>(live_in);
    // TODO: Following logic needs to be refactored.
    for (Block *pred : block->getPredecessors()) {
      Value incoming;
      Value branch_pred;
      Operation *term = pred->getTerminator();
      // If it's a branch or cond_br, get the value passed into this block argument
      if (auto br = dyn_cast<neura::Br>(term)) {
        auto args = br.getArgs();
        if (arg) {
          unsigned arg_index = arg.getArgNumber();
          assert(arg_index < args.size());
          incoming = args[arg_index];
        } else if (live_in.getDefiningOp()->getBlock() == pred) {
          // Handles the case where live_in is not a block argument.
          incoming = live_in;
        } else {
          // If live_in is not a block argument and not defined in the block, skips.
          continue;
        }
      } else if (auto condBr = dyn_cast<neura::CondBr>(term)) {
        Value cond = condBr.getCondition();
        branch_pred = cond; // by default
        OpBuilder pred_builder(condBr);
        Location pred_loc = condBr.getLoc();

        if (condBr.getTrueDest() == block) {
          if (arg) {
            auto trueArgs = condBr.getTrueArgs();
            unsigned arg_index = arg.getArgNumber();
            assert(arg_index < trueArgs.size());
            incoming = trueArgs[arg_index];
          } else if (live_in.getDefiningOp()->getBlock() == pred) {
            // Handles the case where live_in is not a block argument.
            incoming = live_in;
          } else {
            // If live_in is not a block argument and not defined in the block, skips.
            continue;
          }
          // Applies grant_predicate.
          incoming = pred_builder.create<neura::GrantPredicateOp>(
            pred_loc, incoming.getType(), incoming, cond);
          just_created_consumer_ops.insert(incoming.getDefiningOp());
          // Keep branch_pred = cond
        } else if (condBr.getFalseDest() == block) {
          if (arg) {
            auto falseArgs = condBr.getFalseArgs();
            unsigned arg_index = arg.getArgNumber();
            assert(arg_index < falseArgs.size());
            incoming = falseArgs[arg_index];
          } else if (live_in.getDefiningOp()->getBlock() == pred) {
            // Handles the case where live_in is not a block argument.
            incoming = live_in;
          } else {
            // If live_in is not a block argument and not defined in the block, skips.
            continue;
          }
          // Negates cond for false edge.
          branch_pred = pred_builder.create<neura::NotOp>(pred_loc, cond.getType(), cond);
          // Applies grant_predicate.
          incoming = pred_builder.create<neura::GrantPredicateOp>(
            pred_loc, incoming.getType(), incoming, branch_pred);
          just_created_consumer_ops.insert(incoming.getDefiningOp());
        } else {
          llvm::errs() << "cond_br does not target block:\n" << *block << "\n";
          assert(false);
        }
      } else {
        llvm::errs() << "Unknown branch terminator in block: " << *pred << "\n";
        continue;
      }

      // If the incoming value is defined in the same block, inserts a `neura.reserve`
      // and defer a backward ctrl move.
      if (incoming.getDefiningOp() && incoming.getDefiningOp()->getBlock() == block) {
        builder.setInsertionPointToStart(block);
        auto placeholder = builder.create<neura::ReserveOp>(loc, incoming.getType());
        phi_operands.push_back(placeholder.getResult());
        // Defers the backward ctrl move operation to be inserted after all phi operands
        // are defined. Inserted:
        // (real_defined_value, just_created_reserve, branch_pred, current_block).
        deferred_ctrl_movs.emplace_back(
          incoming, placeholder.getResult(), branch_pred, block);
      } else {
        phi_operands.push_back(incoming);
      }
      // If live_in is not a block argument, we don't need to check for uniqueness.
      if (!arg) {
        continue;
      }
    }

    assert(!phi_operands.empty());

    // Puts all operands into a set to ensure uniqueness. Specifically, following
    // case is handled:
    // ---------------------------------------------------------
    // ^bb1:
    //   "neura.br"(%a)[^bb3] : (!neura.data<f32, i1>) -> ()
    //
    // ^bb2:
    //   "neura.br"(%a)[^bb3] : (!neura.data<f32, i1>) -> ()
    //
    // ^bb3(%x: !neura.data<f32, i1>):
    //   ...
    // ---------------------------------------------------------
    // In above case, %a is used in both branches of the control flow, so we
    // don't need a phi node, but we still need to replace its uses with the
    // result of the phi node.
    // This ensures that we only create a phi node if there are multiple unique
    // operands.
    llvm::SmallDenseSet<Value, 4> unique_operands(phi_operands.begin(), phi_operands.end());

    if (unique_operands.size() == 1) {
      // No phi needed, but still replace
      Value single = *unique_operands.begin();
      SmallVector<OpOperand *, 4> uses;
      for (OpOperand &use : live_in.getUses()) {
        // Skip uses that were just created by the grant_predicate.
        if (!just_created_consumer_ops.contains(use.getOwner())) {
          uses.push_back(&use);
        }
      }
      for (OpOperand *use : uses) {
        use->set(single);
      }
      // No need to proceed further to create a phi node, as we have a single unique operand.
      continue;
    }

    // Creates the phi node with dynamic number of operands.
    auto phi_op = builder.create<neura::PhiOp>(loc, predicated_type, phi_operands);

    // Saves users to be replaced *after* phi is constructed.
    SmallVector<OpOperand *> uses_to_be_replaced;
    for (OpOperand &use : live_in.getUses()) {
      if (use.getOwner() != phi_op) {
        uses_to_be_replaced.push_back(&use);
      }
    }
    // Replaces live-in uses with the phi result.
    for (OpOperand *use : uses_to_be_replaced) {
      use->set(phi_op.getResult());
    }
  }
}

namespace {
struct TransformCtrlToDataFlowPass 
    : public PassWrapper<TransformCtrlToDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformCtrlToDataFlowPass)

  StringRef getArgument() const override { return "transform-ctrl-to-data-flow"; }
  StringRef getDescription() const override {
    return "Transforms control flow into data flow using predicated execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Declares a vector to hold deferred backward ctrl move operations.
    // This is useful when a live-in value is defined within the same block.
    // The tuple contains:
    // - real value (the one that is defined in the same block, after the placeholder)
    // - placeholder value (the one that will be used in the phi node)
    // - branch predicate (if any, for cond_br)
    // - block where the backward ctrl move should be inserted
    SmallVector<std::tuple<Value, Value, Value, Block *>, 4> deferred_ctrl_movs;
    module.walk([&](func::FuncOp func) {

      OpBuilder builder(func.getContext());
      DenseMap<Value, Value> granted_once_map;
      insertGrantOnceInEntryBlock(&func.getBody().front(), builder, granted_once_map);

      // Get blocks in post-order
      SmallVector<Block *> postOrder;
      DenseSet<Block *> visited;
      getBlocksInPostOrder(&func.getBody().front(), postOrder, visited);

      // Process blocks bottom-up
      for (Block *block : postOrder) {
        // Creates phi nodes for live-ins.
        createPhiNodesForBlock(block, builder, deferred_ctrl_movs);
      }

      // Flattens blocks into the entry block.
      Block *entryBlock = &func.getBody().front();
      SmallVector<Block *> blocks_to_flatten;
      for (Block &block : func.getBody()) {
        if (&block != entryBlock)
          blocks_to_flatten.push_back(&block);
      }

      // Erases terminators before moving ops into entry block.
      for (Block *block : blocks_to_flatten) {
        for (Operation &op : llvm::make_early_inc_range(*block)) {
          if (isa<neura::Br>(op) || isa<neura::CondBr>(op)) {
            op.erase();
          }
        }
      }

      // Moves all operations from blocks to the entry block before the terminator.
      for (Block *block : blocks_to_flatten) {
        auto &ops = block->getOperations();
        while (!ops.empty()) {
          Operation &op = ops.front();
          op.moveBefore(&entryBlock->back());
        }
      }

      // Erases any remaining br/cond_br that were moved into the entry block.
      for (Operation &op : llvm::make_early_inc_range(*entryBlock)) {
        if (isa<neura::Br>(op) || isa<neura::CondBr>(op)) {
          op.erase();
        }
      }

      for (Block *block : blocks_to_flatten) {
        block->erase();
      }
    });

    // Inserts the deferred backward ctrl move operations after phi operands
    // are defined.
    for (auto &[real_dependent, placeholder, branch_pred, block] : deferred_ctrl_movs) {
      Operation *def_op = real_dependent.getDefiningOp();
      assert(def_op && "Backward ctrl move's source must be an op result");

      // Find the correct insertion point: after both real_dependent and branch_pred
      Operation *insert_after = def_op;
      if (Operation *pred_def = branch_pred.getDefiningOp()) {
        if (insert_after->isBeforeInBlock(pred_def))
          insert_after = pred_def;
      }

      OpBuilder mov_builder(insert_after->getBlock(), ++Block::iterator(insert_after));
      Location insert_loc = insert_after->getLoc();

      Value guarded_val = real_dependent;

      mov_builder.create<neura::CtrlMovOp>(insert_loc, guarded_val, placeholder);
    }
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura