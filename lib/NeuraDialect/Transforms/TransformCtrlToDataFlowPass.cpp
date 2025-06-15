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
void createPhiNodesForBlock(Block *block, OpBuilder &builder,
                            DenseMap<Value, Value> &value_map) {
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
        // Checks if the live-in is a block argument. SSA form forces this rule.
        bool found_in_block_argument = false;
        for (BlockArgument arg : block->getArguments()) {
          if (arg == operand) {
            found_in_block_argument = true;
            break;
          }
        }
        // assert(found_in_block_argument && "Live-in value defined outside the block must be passed as a block argument");
        live_ins.push_back(operand);
      }

      // Collects all block arguments.
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(operand)) {
        live_ins.push_back(operand);
      }
    }
  }

  // Creates a phi node for each live-in value.
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
    BlockArgument arg = dyn_cast<BlockArgument>(live_in);
    // Handles the case where live_in is not a block argument.
    if (!arg) {
      phi_operands.push_back(live_in);
    } else {
      // Finds index of live_in in block arguments.
      unsigned arg_index = arg.getArgNumber();
      for (Block *pred : block->getPredecessors()) {
        Value incoming;
        Operation *term = pred->getTerminator();

        // If it's a branch or cond_br, get the value passed into this block argument
        if (auto br = dyn_cast<neura::Br>(term)) {
          auto args = br.getArgs();
          assert(arg_index < args.size());
          incoming = args[arg_index];
        } else if (auto condBr = dyn_cast<neura::CondBr>(term)) {
          if (condBr.getTrueDest() == block) {
            auto trueArgs = condBr.getTrueArgs();
            assert(arg_index < trueArgs.size());
            incoming = trueArgs[arg_index];
          } else if (condBr.getFalseDest() == block) {
            auto falseArgs = condBr.getFalseArgs();
            assert(arg_index < falseArgs.size());
            incoming = falseArgs[arg_index];
          } else {
            llvm::errs() << "cond_br does not target block:\n" << *block << "\n";
            continue;
          }
        } else {
          llvm::errs() << "Unknown branch terminator in block: " << *pred << "\n";
          continue;
        }
        phi_operands.push_back(incoming);
      }
    }

    assert(!phi_operands.empty());

    // Creates the phi node with dynamic number of operands.
    auto phi_op = builder.create<neura::PhiOp>(loc, predicated_type, phi_operands);

    // Saves users to be replaced *after* phi is constructed.
    SmallVector<OpOperand *> uses_to_be_replaced;
    for (OpOperand &use : live_in.getUses()) {
      if (use.getOwner() != phi_op) {
        uses_to_be_replaced.push_back(&use);
      }
    }
    // Replaces block argument use with the phi result.
    for (OpOperand *use : uses_to_be_replaced) {
      use->set(phi_op.getResult());
    }
    value_map[live_in] = phi_op.getResult();
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

    module.walk([&](func::FuncOp func) {
      // Get blocks in post-order
      SmallVector<Block *> postOrder;
      DenseSet<Block *> visited;
      getBlocksInPostOrder(&func.getBody().front(), postOrder, visited);

      // Value mapping for phi node creation.
      DenseMap<Value, Value> value_map;
      OpBuilder builder(func.getContext());

      // Process blocks bottom-up
      for (Block *block : postOrder) {
        // Creates phi nodes for live-ins.
        createPhiNodesForBlock(block, builder, value_map);
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
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura