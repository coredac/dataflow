#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>

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

// Control flow struct.
struct ControlFlowInfo {
  struct Edge {
    Block *source;
    Block *target;
    Value condition; // Optional condition for the edge.
    bool is_not_condition;
    SmallVector<Value> passed_values; // Values passed to the target block.
    bool is_back_edge;
  };
  SmallVector<std::unique_ptr<Edge>> all_edges; // All edges in the function.
  llvm::MapVector<Block *, SmallVector<Edge *>>
      incoming_edges; // Incoming edges for each block.
  llvm::MapVector<Block *, SmallVector<Edge *>>
      outgoing_edges; // Outgoing edges for each block.

  llvm::MapVector<Block *, SmallVector<Edge *>> back_edges;
  llvm::MapVector<Block *, SmallVector<Edge *>> forward_edges;
  llvm::SmallVector<Block *>
      blocks_with_back_edges; // Blocks with backward edges.

  Edge *createEdge() {
    all_edges.push_back(std::make_unique<Edge>());
    return all_edges.back().get();
  }
};

// Checks if all the live-out values in a block are dominated by the block's
// arguments.
void assertLiveOutValuesDominatedByBlockArgs(Region &region) {
  llvm::errs()
      << "[ctrl2data] Asserting live-out values dominated by block arguments\n";
  for (Block &block : region) {
    if (&block == &region.front()) {
      continue;
    }

    DenseSet<Value> live_out_values;
    for (Operation &op : block) {
      for (Value result : op.getResults()) {
        for (OpOperand &use : result.getUses()) {
          if (use.getOwner()->getBlock() != &block) {
            live_out_values.insert(result);
            break;
          }
        }
      }
    }

    // Skips blocks with no live-out values.
    if (live_out_values.empty())
      continue;

    DenseSet<Value> dominated_values;

    if (block.getNumArguments() == 0 && !live_out_values.empty()) {
      assert(false && "Block without arguments has live-out values");
    }

    for (BlockArgument arg : block.getArguments()) {
      dominated_values.insert(arg);
    }

    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          if (dominated_values.count(result))
            continue;

          for (Value operand : op.getOperands()) {
            if (dominated_values.count(operand)) {
              dominated_values.insert(result);
              changed = true;
              break;
            }
          }
        }
      }
    }
    for (Value live_out : live_out_values) {
      if (!dominated_values.count(live_out)) {
        assert(false && "Live-out value not dominated by block arguments or "
                        "live-in values");
      }
    }
  }

  llvm::errs() << "[ctrl2data] All live-out values are dominated by block "
                  "arguments or live-in values.\n";
}

// Builds control flow info for the given function.
void buildControlFlowInfo(Region &region, ControlFlowInfo &ctrl_info,
                          DominanceInfo &dom_info) {
  for (Block &block : region) {
    Operation *terminator = block.getTerminator();

    if (auto cond_br = dyn_cast<neura::CondBr>(terminator)) {
      Block *true_dest = cond_br.getTrueDest();
      Block *false_dest = cond_br.getFalseDest();

      // Creates an edge for true destination.
      ControlFlowInfo::Edge *true_edge = ctrl_info.createEdge();
      true_edge->source = &block;
      true_edge->target = true_dest;
      true_edge->condition = cond_br.getCondition();
      true_edge->is_not_condition = false;
      true_edge->is_back_edge = dom_info.dominates(true_dest, &block);
      for (Value passed_value : cond_br.getTrueArgs()) {
        true_edge->passed_values.push_back(passed_value);
      }

      // Creates an edge for false destination.
      ControlFlowInfo::Edge *false_edge = ctrl_info.createEdge();
      false_edge->source = &block;
      false_edge->target = false_dest;
      false_edge->condition = cond_br.getCondition();
      false_edge->is_not_condition = true;
      false_edge->is_back_edge = dom_info.dominates(false_dest, &block);
      for (Value passed_value : cond_br.getFalseArgs()) {
        false_edge->passed_values.push_back(passed_value);
      }

      // Creates the blocks to edges mapping.
      ctrl_info.outgoing_edges[&block].push_back(true_edge);
      ctrl_info.outgoing_edges[&block].push_back(false_edge);
      ctrl_info.incoming_edges[true_dest].push_back(true_edge);
      ctrl_info.incoming_edges[false_dest].push_back(false_edge);

      // Handles back edges.
      if (true_edge->is_back_edge) {
        ctrl_info.back_edges[&block].push_back(true_edge);
        ctrl_info.blocks_with_back_edges.push_back(&block);
      } else {
        ctrl_info.forward_edges[&block].push_back(true_edge);
      }
      if (false_edge->is_back_edge) {
        ctrl_info.back_edges[&block].push_back(false_edge);
        ctrl_info.blocks_with_back_edges.push_back(&block);
      } else {
        ctrl_info.forward_edges[&block].push_back(false_edge);
      }

    } else if (auto br = dyn_cast<neura::Br>(terminator)) {
      Block *dest = br.getDest();

      // Creates unconditional edge to the destination block.
      ControlFlowInfo::Edge *edge = ctrl_info.createEdge();
      edge->source = &block;
      edge->target = dest;
      edge->condition = nullptr; // No condition for Br.
      edge->is_not_condition = false;
      edge->is_back_edge = dom_info.dominates(dest, &block);
      for (Value passed_value : br.getArgs()) {
        edge->passed_values.push_back(passed_value);
      }

      // Updates the blocks to edges mapping.
      ctrl_info.outgoing_edges[&block].push_back(edge);
      ctrl_info.incoming_edges[dest].push_back(edge);

      // Handles back edges.
      if (edge->is_back_edge) {
        ctrl_info.back_edges[&block].push_back(edge);
        ctrl_info.blocks_with_back_edges.push_back(&block);
      } else {
        ctrl_info.forward_edges[&block].push_back(edge);
      }

    } else if (auto rt = dyn_cast<neura::ReturnOp>(terminator)) {
      llvm::errs() << "[ctrl2data] ReturnOp found: " << *rt << "\n";
    } else {
      llvm::errs() << "[ctrl2data] Unknown terminator: " << *terminator << "\n";
      assert(false);
    }
  }
}

Value getPrecessedCondition(Value condition, bool is_not_condition,
                            llvm::MapVector<Value, Value> &condition_cache,
                            OpBuilder &builder) {
  if (!is_not_condition) {
    return condition;
  }

  auto it = condition_cache.find(condition);
  if (it != condition_cache.end()) {
    return it->second;
  }

  Block *source = condition.getDefiningOp()->getBlock();
  builder.setInsertionPoint(source->getTerminator());
  Value not_condition = builder.create<neura::NotOp>(
      condition.getLoc(), condition.getType(), condition);
  condition_cache[condition] = not_condition;
  return not_condition;
}

void createReserveAndPhiOps(
    Region &region, ControlFlowInfo &ctrl_info,
    llvm::MapVector<BlockArgument, Value> &arg_to_reserve,
    llvm::MapVector<BlockArgument, Value> &arg_to_phi_result,
    OpBuilder &builder) {
  DominanceInfo dom_info(region.getParentOp());

  // ================================================
  // Step 1: Categorizes edges into six types.
  // ================================================
  // Type 1: Backward cond_br edges with arguments.
  // Type 2: Backward br edges with values.
  // Type 3: Forward cond_br edges with values.
  // Type 4: Forward br edges with values.
  // Type 5: Forward cond_br edges without values.
  // Type 6: Forward br edges without values.
  // For Backward edges without values, they can be transformed into type 1 or 2

  // Uses llvm::MapVector instead of DenseMap to maintain insertion order.
  llvm::MapVector<BlockArgument, SmallVector<ControlFlowInfo::Edge *>>
      backward_value_edges;
  llvm::MapVector<BlockArgument, SmallVector<ControlFlowInfo::Edge *>>
      forward_value_edges;
  llvm::MapVector<Block *, SmallVector<ControlFlowInfo::Edge *>>
      block_conditional_edges;

  llvm::MapVector<Value, Value> condition_cache;

  llvm::MapVector<BlockArgument, SmallVector<Value>> arg_to_phi_operands;

  // Tracks the mapping of live-out values.
  llvm::MapVector<Value, Value> value_to_predicated_value;

  for (auto &edge : ctrl_info.all_edges) {
    Block *target = edge->target;

    // Type 1 & 2: Backward cond_br/br edges with values.
    if (edge->is_back_edge && !edge->passed_values.empty()) {
      if (edge->passed_values.size() != target->getNumArguments()) {
        llvm::errs()
            << "[ctrl2data] Error: Number of passed values does not match "
               "target block arguments.\n";
        assert(false);
      }
      for (BlockArgument arg : target->getArguments()) {
        backward_value_edges[arg].push_back(edge.get());
      }
    }
    // Type 3 & 4: Forward cond_br/br edges with values.
    else if (!edge->is_back_edge && !edge->passed_values.empty()) {
      ;
      if (edge->passed_values.size() != target->getNumArguments()) {
        llvm::errs()
            << "[ctrl2data] Error: Number of passed values does not match "
               "target block arguments.\n";
        assert(false);
      }
      for (BlockArgument arg : target->getArguments()) {
        forward_value_edges[arg].push_back(edge.get());
      }
    }
    // Type 5 & 6: Forward cond_br/br edges without values.
    else if (!edge->is_back_edge && edge->passed_values.empty()) {
      if (edge->condition) {
        // Only Type 5 needs to be processed here.
        // If the edge has a condition, store it for later use.
        block_conditional_edges[target].push_back(edge.get());
      }
    }
  }

  // ================================================
  // Step 2: Handles Forward cond_br edges without values.
  // ================================================
  // Handles Type 5 edges.
  for (auto &condition_pair : block_conditional_edges) {
    Block *target = condition_pair.first;
    auto &edges = condition_pair.second;

    if (edges.empty()) {
      continue;
    }

    // Collects all conditions for the target block.
    SmallVector<Value> conditions;
    for (ControlFlowInfo::Edge *edge : edges) {
      Value condition = getPrecessedCondition(
          edge->condition, edge->is_not_condition, condition_cache, builder);
      conditions.push_back(condition);
    }

    // Unsupported case: multiple conditions for a single block.
    // TODO: Adds support if needed.
    if (conditions.size() > 1) {
      llvm::errs() << "[ctrl2data] Unsupported case: multiple conditions for a "
                      "single block: "
                   << *target << "\n";
      assert(false);
    }

    if (target->getArguments().empty()) {
      // Grants predicate for all the live-in values in the target block.
      // Uses SetVector instead of DenseSet to maintain insertion order.
      SetVector<Value> live_in_values;
      for (Operation &op : target->getOperations()) {
        for (Value operand : op.getOperands()) {
          if (operand.getDefiningOp() &&
              operand.getDefiningOp()->getBlock() != target &&
              !isa<neura::ReserveOp>(operand.getDefiningOp())) {
            live_in_values.insert(operand);
          }
        }
      }

      // Applies grant_predicate for each live-in value.
      for (Value live_in_value : live_in_values) {
        // Finds the earliest use of the live-in value.
        Operation *earliest_use = nullptr;
        for (Operation &op : target->getOperations()) {
          for (Value operand : op.getOperands()) {
            if (operand == live_in_value) {
              earliest_use = &op;
              break;
            }
          }
          if (earliest_use) {
            break;
          }
        }

        if (earliest_use) {
          builder.setInsertionPoint(earliest_use);
        } else {
          builder.setInsertionPointToStart(target);
        }

        // Creates predicated version of the live-in value.
        Value predicated_value = builder.create<neura::GrantPredicateOp>(
            live_in_value.getLoc(), live_in_value.getType(), live_in_value,
            conditions[0]);

        value_to_predicated_value[live_in_value] = predicated_value;

        // Replace uses of the live-in value within this block only.
        for (OpOperand &use :
             llvm::make_early_inc_range(live_in_value.getUses())) {
          if (use.getOwner()->getBlock() == target &&
              use.getOwner() != predicated_value.getDefiningOp()) {
            use.set(predicated_value);
          }
        }
      }
    }
  }

  // Updates the passed values in edges with predicated values.
  for (auto &edge_ptr : ctrl_info.all_edges) {
    ControlFlowInfo::Edge *edge = edge_ptr.get();
    for (size_t i = 0; i < edge->passed_values.size(); ++i) {
      Value val = edge->passed_values[i];
      if (value_to_predicated_value.count(val)) {
        edge->passed_values[i] = value_to_predicated_value[val];
      }
    }
  }

  // ================================================
  // Step 3: Creates reserve and ctrl_mov operations for needed blockarguments.
  // ================================================
  // Handles Type 1 & 2 edges.
  for (auto &backward_pair : backward_value_edges) {
    BlockArgument arg = backward_pair.first;
    auto &edges = backward_pair.second;
    Block *block = arg.getOwner();
    builder.setInsertionPointToStart(block);
    neura::ReserveOp reserve_op =
        builder.create<neura::ReserveOp>(arg.getLoc(), arg.getType());
    arg_to_reserve[arg] = reserve_op.getResult();
    arg_to_phi_operands[arg].push_back(reserve_op.getResult());

    // Creates ctrl_mov operations for reserved values.
    for (ControlFlowInfo::Edge *edge : edges) {
      Value val = edge->passed_values[arg.getArgNumber()];
      builder.setInsertionPoint(edge->source->getTerminator());

      Value predicated_val = val;
      if (edge->condition) {
        Value processed_condition = getPrecessedCondition(
            edge->condition, edge->is_not_condition, condition_cache, builder);
        predicated_val = builder.create<neura::GrantPredicateOp>(
            edge->condition.getLoc(), val.getType(), predicated_val,
            processed_condition);
      }

      // Creates ctrl_mov operation.
      builder.create<neura::CtrlMovOp>(val.getLoc(), predicated_val,
                                       reserve_op);
    }
  }

  // ================================================
  // Step 4: Prepares for creating phi operations.
  // ================================================
  // Handles Type 3 & 4 edges.

  for (auto &forward_pair : forward_value_edges) {
    BlockArgument arg = forward_pair.first;
    auto &edges = forward_pair.second;

    for (ControlFlowInfo::Edge *edge : edges) {
      Value val = edge->passed_values[arg.getArgNumber()];

      Value predicated_val = val;
      if (edge->condition) {
        builder.setInsertionPoint(edge->source->getTerminator());
        Value processed_condition = getPrecessedCondition(
            edge->condition, edge->is_not_condition, condition_cache, builder);

        predicated_val = builder.create<neura::GrantPredicateOp>(
            edge->condition.getLoc(), predicated_val.getType(), predicated_val,
            processed_condition);
      }

      arg_to_phi_operands[arg].push_back(predicated_val);
    }
  }

  // ================================================
  // Step 5: Creates phi operations for each block argument.
  // ================================================
  for (auto &arg_to_phi_pair : arg_to_phi_operands) {
    BlockArgument arg = arg_to_phi_pair.first;
    auto &phi_operands = arg_to_phi_pair.second;

    if (phi_operands.size() <= 1) {
      // No need to create a phi operation if there's only one operand.

      if (phi_operands.size() == 1) {
        arg_to_phi_result[arg] = phi_operands[0];
        arg.replaceAllUsesWith(phi_operands[0]);
      }
      continue;
    }

    // Handles the blcockargument with/without reserve seperately (different
    // insertion points).
    if (arg_to_reserve.count(arg)) {
      Value reserve_value = arg_to_reserve[arg];
      builder.setInsertionPointAfter(reserve_value.getDefiningOp());

      // Creates phi operation for reserved values.
      auto phi = builder.create<neura::PhiOp>(arg.getLoc(), arg.getType(),
                                              phi_operands);
      arg_to_phi_result[arg] = phi;
    } else {
      Block *placement = arg.getParentBlock();

      builder.setInsertionPointToStart(placement);

      // Creates phi operation for block argument without reserve.
      auto phi = builder.create<neura::PhiOp>(arg.getLoc(), arg.getType(),
                                              phi_operands);
      arg.replaceAllUsesWith(phi);
      arg_to_phi_result[arg] = phi;
    }
  }
}

// Transforms control flow into data flow.
void transformControlFlowToDataFlow(Region &region, ControlFlowInfo &ctrl_info,
                                    DominanceInfo &dom_info,
                                    OpBuilder &builder) {

  // Asserts that all live-out values are dominated by block arguments.
  assertLiveOutValuesDominatedByBlockArgs(region);

  // Creates reserve and phi operations for each block argument.
  llvm::MapVector<BlockArgument, Value> arg_to_reserve;
  llvm::MapVector<BlockArgument, Value> arg_to_phi_result;
  createReserveAndPhiOps(region, ctrl_info, arg_to_reserve, arg_to_phi_result,
                         builder);

  // Replaces blockarguments with phi results
  for (auto &arg_to_phi_pair : arg_to_phi_result) {
    BlockArgument arg = arg_to_phi_pair.first;
    Value phi_result = arg_to_phi_pair.second;
    arg.replaceAllUsesWith(phi_result);
  }

  // Flattens blocks into the entry block.
  Block *entryBlock = &region.front();
  SmallVector<Block *> blocks_to_flatten;
  for (Block &block : region) {
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

  // Erases now-empty blocks
  for (Block *block : blocks_to_flatten) {
    block->erase();
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
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      Region *region = nullptr;
      DominanceInfo domInfo;
      OpBuilder builder(op->getContext());

      if (auto func = dyn_cast<func::FuncOp>(op)) {
        auto accel_attr = func->getAttrOfType<StringAttr>("accelerator");
        if (!accel_attr || accel_attr.getValue() != "neura") {
          return;
        }
        region = &func.getBody();
        domInfo = DominanceInfo(func);
        GrantPredicateInEntryBlock(&region->front(), builder);
        assertLiveOutValuesDominatedByBlockArgs(*region);
      } else if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        if (llvmFunc.isDeclaration()) {
          return;
        }
        auto accel_attr = llvmFunc->getAttrOfType<StringAttr>("accelerator");
        if (!accel_attr || accel_attr.getValue() != "neura") {
          return;
        }
        region = &llvmFunc.getBody();
        domInfo = DominanceInfo(llvmFunc);
        GrantPredicateInEntryBlock(&region->front(), builder);
        assertLiveOutValuesDominatedByBlockArgs(*region);
      } else {
        return;
      }
      ControlFlowInfo ctrlInfo;
      buildControlFlowInfo(*region, ctrlInfo, domInfo);
      transformControlFlowToDataFlow(*region, ctrlInfo, domInfo, builder);
    });
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura