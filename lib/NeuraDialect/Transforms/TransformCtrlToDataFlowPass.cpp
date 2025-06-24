#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define GEN_PASS_DEF_TransformCtrlToDataFlow
#include "NeuraDialect/NeuraPasses.h.inc"

// Inserts `grant_once` for every predicated value defined in the entry block
// that is used outside of the block (i.e., a live-out).
void GrantPredicateInEntryBlock(Block *entry_block, OpBuilder &builder) {
  SmallVector<Value> live_out_arg_values;
  SmallVector<Value> live_out_non_arg_values;

  // Step 1: Collects all live-out values first.
  for (Operation &op : *entry_block) {
    for (Value result : op.getResults()) {
      if (!isa<neura::PredicatedValue>(result.getType()))
        continue;

      bool used_in_branch = false;
      bool used_elsewhere = false;

      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();

        // Case 1: Operand of a branch/cond_br → grant_once
        if (isa<neura::Br, neura::CondBr>(user)) {
          used_in_branch = true;
        }

        // Case 2: Used directly in other blocks → grant_always
        if (user->getBlock() != entry_block) {
          used_elsewhere = true;
        }
      }

      if (used_in_branch)
        live_out_arg_values.push_back(result);
      if (used_elsewhere)
        live_out_non_arg_values.push_back(result);
    }
  }

  // Step 2: Inserts grant_once for each candidate.
  // Inserts grant_once.
  for (Value val : live_out_arg_values) {
    Operation *def_op = val.getDefiningOp();
    if (!def_op)
      continue;

    builder.setInsertionPointAfter(def_op);
    auto granted = builder.create<neura::GrantOnceOp>(def_op->getLoc(),
                                                      val.getType(), val);

    // Replaces uses in branch ops.
    for (OpOperand &use : llvm::make_early_inc_range(val.getUses())) {
      Operation *user = use.getOwner();
      if (isa<neura::Br, neura::CondBr>(user)) {
        use.set(granted.getResult());
      }
    }
  }

  // Inserts grant_always.
  for (Value val : live_out_non_arg_values) {
    Operation *def_op = val.getDefiningOp();
    if (!def_op)
      continue;

    builder.setInsertionPointAfter(def_op);
    auto granted = builder.create<neura::GrantAlwaysOp>(def_op->getLoc(),
                                                        val.getType(), val);

    // Replaces direct external uses (not in entry block, not in branch ops).
    for (OpOperand &use : llvm::make_early_inc_range(val.getUses())) {
      Operation *user = use.getOwner();
      if (user->getBlock() != entry_block &&
          !isa<neura::Br, neura::CondBr>(user)) {
        use.set(granted.getResult());
      }
    }
  }
}

// Returns blocks in post-order traversal order.
void getBlocksInPostOrder(Block *startBlock,
                          SmallVectorImpl<Block *> &postOrder,
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
    Block *block, Block *entry_block, OpBuilder &builder,
    SmallVectorImpl<std::tuple<Value, Value, Value, Block *>>
        &deferred_ctrl_movs) {
  if (block->hasNoPredecessors()) {
    // Skips phi insertion for entry block.
    return;
  }

  bool has_block_args = false;
  // Collects all live-in values.
  SmallVector<Value> live_ins;
  for (Operation &op : *block) {
    for (Value operand : op.getOperands()) {
      // Identifies operands defined in other blocks.
      if (operand.getDefiningOp() &&
          operand.getDefiningOp()->getBlock() != block) {
        if (!llvm::is_contained(live_ins, operand)) {
          live_ins.push_back(operand);
          continue;
        }
      }

      // Collects block arguments as live-ins.
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (!llvm::is_contained(live_ins, arg) && arg.getOwner() == block) {
          has_block_args = true;
          live_ins.push_back(arg);
          continue;
        }
      }
    }
  }

  builder.setInsertionPointToStart(block);
  // Uses the location from the first operation in the block or block's parent
  // operation.
  Location loc =
      block->empty() ? block->getParent()->getLoc() : block->front().getLoc();
  if (has_block_args) {
    for (Value live_in : live_ins) {
      // Creates predicated type for phi node.
      Type live_in_type = live_in.getType();
      Type predicated_type =
          isa<neura::PredicatedValue>(live_in_type)
              ? live_in_type
              : neura::PredicatedValue::get(builder.getContext(), live_in_type,
                                            builder.getI1Type());

      BlockArgument block_arg = dyn_cast<BlockArgument>(live_in);
      bool is_block_arg = block_arg && block_arg.getOwner() == block;

      if (is_block_arg) {
        SmallVector<Value> phi_operands;
        llvm::SmallDenseSet<Operation *, 4> just_created_consumer_ops;
        for (Block *pred : block->getPredecessors()) {
          Value incoming_in_pred;
          Operation *term = pred->getTerminator();
          if (neura::Br br = dyn_cast<neura::Br>(term)) {
            auto pred_br_args = br.getArgs();
            unsigned arg_index = block_arg.getArgNumber();
            assert(arg_index < pred_br_args.size() && "Invalid arg index");
            incoming_in_pred = pred_br_args[arg_index];
          } else if (neura::CondBr cond_br = dyn_cast<neura::CondBr>(term)) {
            Value cond = cond_br.getCondition();
            OpBuilder pred_builder(cond_br);
            Location pred_loc = cond_br.getLoc();

            if (cond_br.getTrueDest() == block) {
              auto pred_true_args = cond_br.getTrueArgs();
              unsigned arg_index = block_arg.getArgNumber();
              assert(arg_index < pred_true_args.size() && "Invalid arg index");
              incoming_in_pred = pred_true_args[arg_index];

              // Applies grant_predicate.
              incoming_in_pred = pred_builder.create<neura::GrantPredicateOp>(
                  pred_loc, incoming_in_pred.getType(), incoming_in_pred, cond);
              just_created_consumer_ops.insert(
                  incoming_in_pred.getDefiningOp());
            } else if (cond_br.getFalseDest() == block) {
              auto pred_false_args = cond_br.getFalseArgs();
              unsigned arg_index = block_arg.getArgNumber();
              assert(arg_index < pred_false_args.size() && "Invalid arg index");
              incoming_in_pred = pred_false_args[arg_index];

              // Negates cond for false edge.
              Value not_cond = pred_builder.create<neura::NotOp>(
                  pred_loc, cond.getType(), cond);
              // Applies grant_predicate.
              incoming_in_pred = pred_builder.create<neura::GrantPredicateOp>(
                  pred_loc, incoming_in_pred.getType(), incoming_in_pred,
                  not_cond);
              just_created_consumer_ops.insert(
                  incoming_in_pred.getDefiningOp());
            }
          } else {
            llvm::errs() << "[ctrl2data] Unknown branch terminator in block: "
                         << *pred << "\n";
            continue;
          }

          DominanceInfo dom_info(block->getParentOp());
          if (incoming_in_pred.getDefiningOp() &&
              dom_info.dominates(
                  block, incoming_in_pred.getDefiningOp()->getBlock())) {
            builder.setInsertionPointToStart(block);
            Value placeholder = builder.create<neura::ReserveOp>(
                loc, incoming_in_pred.getType());
            phi_operands.push_back(placeholder);
            // Defers the backward ctrl move operation to be inserted after
            // phi operands are defined. Inserted: (real_defined_value,
            // just_created_reserve, branch_pred, current_block).
            deferred_ctrl_movs.emplace_back(incoming_in_pred, placeholder,
                                            nullptr, block);
          } else {
            // No backward dependency found, just add the incoming value.
            phi_operands.push_back(incoming_in_pred);
          }
        }

        // Puts all operands into a set to ensure uniqueness. Specifically,
        // following case is handled:
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
        // In above case, %a is used in both branches of the control flow, so
        // we don't need a phi node, but we still need to replace its uses
        // with the result of the phi node. This ensures that we only create a
        // phi node if there are multiple unique operands.
        SmallVector<Value> unique_operands(phi_operands.begin(),
                                           phi_operands.end());
        if (unique_operands.size() > 1) {
          auto phi_op =
              builder.create<neura::PhiOp>(loc, predicated_type, phi_operands);
          SmallVector<OpOperand *> uses;
          for (OpOperand &use : live_in.getUses()) {
            if (use.getOwner() != phi_op) {
              uses.push_back(&use);
            }
          }
          for (OpOperand *use : uses) {
            use->set(phi_op.getResult());
          }
        } else if (unique_operands.size() == 1) {
          // No phi needed, but still replace
          Value single = unique_operands.front();
          SmallVector<OpOperand *> uses;
          for (OpOperand &use : live_in.getUses()) {
            // Skips uses that were just created by the grant_predicate.
            if (!just_created_consumer_ops.contains(use.getOwner())) {
              uses.push_back(&use);
            }
          }
          for (OpOperand *use : uses) {
            use->set(single);
          }
          continue; // No need to create a phi node.
        }
      }
    }

  } else {
    if (block->hasOneUse() && block->getSinglePredecessor()) {
      Block *pred_block = block->getSinglePredecessor();
      Operation *pred_term = pred_block->getTerminator();
      Value cond;
      if (auto pred_cond_br = dyn_cast<neura::CondBr>(pred_term)) {
        if (pred_cond_br.getTrueDest() == block) {
          cond = pred_cond_br.getCondition();
        } else if (pred_cond_br.getFalseDest() == block) {
          OpBuilder pred_builder(pred_cond_br);
          Location pred_loc = pred_cond_br.getLoc();
          cond = pred_builder.create<neura::NotOp>(
              pred_loc, pred_cond_br.getCondition().getType(),
              pred_cond_br.getCondition());
        }

        SmallVector<std::pair<Operation *, Value>> ops_to_process;
        for (Operation &op : *block) {
          if (isa<neura::Br, neura::CondBr>(op)) {
            continue; // Skips branch ops
          }

          for (Value result : op.getResults()) {
            if (!isa<neura::PredicatedValue>(result.getType()))
              continue;
            ops_to_process.emplace_back(&op, result);
          }
        }

        for (auto [op, result] : ops_to_process) {
          builder.setInsertionPointAfter(op);
          auto predicated = builder.create<neura::GrantPredicateOp>(
              loc, result.getType(), result, cond);
          for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
            if (use.getOwner() != predicated.getOperation()) {
              // Replaces use with the predicated value.
              use.set(predicated.getResult());
            }
          }
        }
      } else {
        llvm::errs()
            << "[ctrl2data] Block has a single predecessor, but it's not a "
               "cond_br: "
            << *pred_term << "\n";
        llvm::errs()
            << "[ctrl2data] Unsupported case, skipping phi node creation.\n";
        return;
      }
    } else {
      llvm::errs()
          << "[ctrl2data] Block has no block arguments and is not a single "
             "predecessor block, skipping phi node creation.\n";
      llvm::errs()
          << "[ctrl2data] Unsupported case, skipping phi node creation.\n";
      return;
    }
  }
}

namespace {
struct TransformCtrlToDataFlowPass
    : public PassWrapper<TransformCtrlToDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformCtrlToDataFlowPass)

  StringRef getArgument() const override {
    return "transform-ctrl-to-data-flow";
  }
  StringRef getDescription() const override {
    return "Transforms control flow into data flow using predicated "
           "execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Declares a vector to hold deferred backward ctrl move operations.
    // This is useful when a live-in value is defined within the same block.
    // The tuple contains:
    // - real value (the one that is defined in the same block, after the
    // placeholder)
    // - placeholder value (the one that will be used in the phi node)
    // - branch predicate (if any, for cond_br)
    // - block where the backward ctrl move should be inserted
    SmallVector<std::tuple<Value, Value, Value, Block *>, 4> deferred_ctrl_movs;
    module.walk([&](func::FuncOp func) {
      OpBuilder builder(func.getContext());
      GrantPredicateInEntryBlock(&func.getBody().front(), builder);

      // Get blocks in post-order
      SmallVector<Block *> postOrder;
      DenseSet<Block *> visited;
      getBlocksInPostOrder(&func.getBody().front(), postOrder, visited);

      // Process blocks bottom-up
      for (Block *block : postOrder) {
        // Creates phi nodes for live-ins.
        createPhiNodesForBlock(block, &func.getBody().front(), builder,
                               deferred_ctrl_movs);
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

      // Moves all operations from blocks to the entry block before the
      // terminator.
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
    for (auto &[real_dependent, placeholder, branch_pred, block] :
         deferred_ctrl_movs) {
      Operation *def_op = real_dependent.getDefiningOp();
      assert(def_op && "Backward ctrl move's source must be an op result");

      // Finds the correct insertion point: after both real_dependent and
      // branch_pred.
      Operation *insert_after = def_op;

      OpBuilder mov_builder(insert_after->getBlock(),
                            ++Block::iterator(insert_after));
      Location insert_loc = insert_after->getLoc();

      Value guarded_val = real_dependent;

      mov_builder.create<neura::CtrlMovOp>(insert_loc, guarded_val,
                                           placeholder);
    }
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura