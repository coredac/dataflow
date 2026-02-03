#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "TaskflowDialect/TaskflowTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//==============================================================================
// Static Affine Loop Tree (SALT) Node.
//==============================================================================
struct SALTNode {
  affine::AffineForOp loop_op;
  int64_t lower_bound;
  int64_t upper_bound;
  int64_t step;

  SALTNode *parent = nullptr;
  SmallVector<SALTNode *> children;

  // Operations that are NOT nested loops (the actual computation at this
  // level).
  SmallVector<Operation *> body_operations;

  bool isLeaf() const { return children.empty(); }
  bool isRoot() const { return parent == nullptr; }
};

//==============================================================================
// Loop Chain - Path from Root to Leaf.
//==============================================================================
struct LoopChain {
  SmallVector<SALTNode *> nodes; // Ordered from root to leaf.

  SALTNode *getRoot() const { return nodes.front(); }
  SALTNode *getLeaf() const { return nodes.back(); }
};

//==============================================================================
// SALT Builder.
//==============================================================================
class SALTBuilder {
public:
  SmallVector<SALTNode *> build(func::FuncOp func_op) {
    SmallVector<SALTNode *> roots;

    for (Block &block : func_op.getBlocks()) {
      for (Operation &op : block) {
        if (affine::AffineForOp for_op = dyn_cast<affine::AffineForOp>(&op)) {
          if (for_op.hasConstantLowerBound() &&
              for_op.hasConstantUpperBound()) {
            SALTNode *root = buildNodeRecursively(for_op, nullptr);
            if (root) {
              roots.push_back(root);
            }
          }
        }
      }
    }

    return roots;
  }

  const SmallVector<std::unique_ptr<SALTNode>> &getAllNodes() const {
    return all_nodes;
  }

private:
  SmallVector<std::unique_ptr<SALTNode>> all_nodes;

  SALTNode *buildNodeRecursively(affine::AffineForOp for_op, SALTNode *parent) {
    auto node = std::make_unique<SALTNode>();
    node->loop_op = for_op;
    node->lower_bound = for_op.getConstantLowerBound();
    node->upper_bound = for_op.getConstantUpperBound();
    node->step = for_op.getStepAsInt();
    node->parent = parent;

    SALTNode *node_ptr = node.get();
    all_nodes.push_back(std::move(node));

    Block &body = for_op.getRegion().front();
    for (Operation &op : body) {
      if (auto nested_for = dyn_cast<affine::AffineForOp>(&op)) {
        if (nested_for.hasConstantLowerBound() &&
            nested_for.hasConstantUpperBound()) {
          SALTNode *child = buildNodeRecursively(nested_for, node_ptr);
          if (child) {
            node_ptr->children.push_back(child);
          }
        } else {
          node_ptr->body_operations.push_back(&op);
        }
      } else if (!isa<affine::AffineYieldOp>(&op)) {
        node_ptr->body_operations.push_back(&op);
      }
    }

    return node_ptr;
  }
};

//==============================================================================
// Loop Chain Extractor (DFS).
//==============================================================================
class LoopChainExtractor {
public:
  SmallVector<LoopChain> extract(const SmallVector<SALTNode *> &roots) {
    SmallVector<LoopChain> chains;

    for (SALTNode *root : roots) {
      SmallVector<SALTNode *> current_path;
      dfs(root, current_path, chains);
    }

    return chains;
  }

private:
  void dfs(SALTNode *node, SmallVector<SALTNode *> &current_path,
           SmallVector<LoopChain> &chains) {
    current_path.push_back(node);

    if (node->isLeaf()) {
      LoopChain chain;
      chain.nodes = current_path;
      chains.push_back(chain);
    } else {
      for (SALTNode *child : node->children) {
        dfs(child, current_path, chains);
      }
    }

    current_path.pop_back();
  }
};

//==============================================================================
// MCT Builder - Builds nested affine.for loops for the entire chain.
//==============================================================================
class MCTBuilder {
public:
  MCTBuilder(OpBuilder &builder, Location loc) : builder(builder), loc(loc) {}

  // Builds the loop chain and returns the outermost loop.
  // The built loops will be inserted at the builder's current insertion point.
  affine::AffineForOp build(const LoopChain &chain) {
    // Mapping from old values to new values.
    IRMapping mapping;

    affine::AffineForOp outer_loop = nullptr;
    Block *current_insert_block = nullptr;
    SmallVector<affine::AffineForOp> created_loops;

    // Iterate from root to leaf to build the nested loops.
    for (size_t i = 0; i < chain.nodes.size(); ++i) {
      SALTNode *node = chain.nodes[i];
      bool is_first = (i == 0);

      OpBuilder loop_builder(builder.getContext());
      if (is_first) {
        loop_builder = builder;
      } else {
        // We want to insert the nested loop at the end of the current block.
        // If the block has a terminator (e.g. yield we just added/cloned?),
        // we should insert before it?
        // Actually, we remove default yield immediately after creation.
        // So the block usually doesn't have a terminator when we are filling
        // it, UNLESS we cloned a yield from body ops? SALT excludes Yields from
        // body_operations. So current_insert_block should be terminator-free
        // (or we removed it).
        loop_builder = OpBuilder::atBlockEnd(current_insert_block);
      }

      // Prepares iter_args for the new loop.
      SmallVector<Value> iter_args_init_values;
      if (node->loop_op.getNumIterOperands() > 0) {
        for (Value init : node->loop_op.getInits()) {
          iter_args_init_values.push_back(mapping.lookupOrDefault(init));
        }
      }

      // Creates new loop with same bounds and iter_args.
      auto new_loop = loop_builder.create<affine::AffineForOp>(
          loc, node->lower_bound, node->upper_bound, node->step,
          iter_args_init_values);

      created_loops.push_back(new_loop);

      // Maps the old induction variable to the new one.
      mapping.map(node->loop_op.getInductionVar(), new_loop.getInductionVar());

      // Maps the old iter_args (block args) to the new iter_args (block args).
      if (node->loop_op.getNumRegionIterArgs() > 0) {
        for (auto [old_arg, new_arg] :
             llvm::zip(node->loop_op.getRegionIterArgs(),
                       new_loop.getRegionIterArgs())) {
          mapping.map(old_arg, new_arg);
        }
      }

      if (is_first) {
        outer_loop = new_loop;
      }

      // Updates current insertion block to the body of the new loop.
      current_insert_block = new_loop.getBody();

      // Removes the default yield created by create<AffineForOp>.
      if (!current_insert_block->empty() &&
          isa<affine::AffineYieldOp>(current_insert_block->back()))
        current_insert_block->back().erase();

      // Clones body operations for THIS node.
      OpBuilder body_builder = OpBuilder::atBlockEnd(current_insert_block);
      for (Operation *op : node->body_operations) {
        Operation *new_op = body_builder.clone(*op, mapping);
        // Updates mapping with results of the new op.
        for (auto [old_res, new_res] :
             llvm::zip(op->getResults(), new_op->getResults())) {
          mapping.map(old_res, new_res);
        }
      }
    }

    // Fixes up yields for non-leaf loops (bottom-up).
    for (int i = created_loops.size() - 2; i >= 0; --i) {
      affine::AffineForOp parent = created_loops[i];
      affine::AffineForOp child = created_loops[i + 1];

      OpBuilder yield_builder = OpBuilder::atBlockEnd(parent.getBody());

      if (child.getNumResults() > 0) {
        yield_builder.create<affine::AffineYieldOp>(loc, child.getResults());
      } else {
        yield_builder.create<affine::AffineYieldOp>(loc);
      }
    }

    // For the LEAF loop, we cloned body operations (which excludes Yields).
    // So the leaf loop likely has NO yield now.
    // We must add a yield to the leaf loop that yields the results of the
    // operations that produced results (mapped from original yield).
    // SALTNode loop_op is the original loop.
    // The original loop body had a yield.
    // We need to find what the original yield yielded, map it, and yield it
    // here.

    // If SALT excludes Yield from body_operations, then we NEVER cloned
    // the yield. So the leaf loop has no terminator. We must reconstruct the
    // yield for the leaf loop.

    if (!created_loops.empty()) {
      affine::AffineForOp new_leaf = created_loops.back();
      SALTNode *leaf_node = chain.getLeaf(); // or chain.nodes.back()

      // Finds the yield op in the original leaf node.
      Operation *original_yield = nullptr;
      for (Operation &op : leaf_node->loop_op.getBody()->getOperations()) {
        if (isa<affine::AffineYieldOp>(&op)) {
          original_yield = &op;
          break;
        }
      }

      if (original_yield) {
        OpBuilder leaf_yield_builder =
            OpBuilder::atBlockEnd(new_leaf.getBody());
        SmallVector<Value> yielded_values;
        for (Value operand : original_yield->getOperands()) {
          yielded_values.push_back(mapping.lookupOrDefault(operand));
        }
        leaf_yield_builder.create<affine::AffineYieldOp>(loc, yielded_values);
      } else {
        assert(false &&
               "Original leaf loop must have a yield operation in its body.");
      }
    }

    return outer_loop;
  }

private:
  OpBuilder &builder;
  Location loc;
};

//==============================================================================
// Pass Implementation
//==============================================================================
struct AffineLoopTreeSerializationPass
    : public PassWrapper<AffineLoopTreeSerializationPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoopTreeSerializationPass)

  StringRef getArgument() const final {
    return "affine-loop-tree-serialization";
  }

  StringRef getDescription() const final {
    return "Serialize Affine loop trees into a linear sequence of loop nests "
           "for MCT construction.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TaskflowDialect, affine::AffineDialect, func::FuncDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    WalkResult result = module.walk([&](func::FuncOp func_op) {
      if (failed(convertFunction(func_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

private:
  LogicalResult convertFunction(func::FuncOp func_op) {
    Location loc = func_op.getLoc();

    // Builds static affine loop tree.
    SALTBuilder salt_builder;
    SmallVector<SALTNode *> roots = salt_builder.build(func_op);

    if (roots.empty()) {
      return success();
    }

    llvm::errs() << "=== SALT Structure ===\n";
    for (SALTNode *root : roots) {
      printSALT(root, 0);
    }

    // Extracts loop chains.
    LoopChainExtractor extractor;
    SmallVector<LoopChain> chains = extractor.extract(roots);

    llvm::errs() << "=== Extracted " << chains.size() << " MCT(s) ===\n";
    for (size_t i = 0; i < chains.size(); ++i) {
      llvm::errs() << "MCT " << i << ": ";
      for (SALTNode *node : chains[i].nodes) {
        llvm::errs() << "[" << node->lower_bound << "," << node->upper_bound
                     << ") ";
      }
      llvm::errs() << "\n";
    }

    // LoopChainExtractor iterates roots in order of SALTBuilder (order
    // of appearance). So we can iterate through roots, and for each root, build
    // its chains, replace root with chains.

    for (SALTNode *root : roots) {
      OpBuilder builder(root->loop_op);

      // Finds chains originating from this root.
      SmallVector<LoopChain> root_chains;
      for (const auto &chain : chains) {
        if (chain.getRoot() == root) {
          root_chains.push_back(chain);
        }
      }

      // Builds new chains.
      for (const LoopChain &chain : root_chains) {
        MCTBuilder mct_builder(builder, loc);
        affine::AffineForOp new_loop = mct_builder.build(chain);

        // If the original root loop had results (iter_args), and the new loop
        // has matching results, we must replace the uses of the original
        // results with the new ones. NOTE: This assumes that for a loop
        // defining values, there is a corresponding single chain that produces
        // all the values (or at least the one we process). If a root with
        // results is split into multiple chains, this simple logic might loop
        // over them. However, for a reduction loop that is a single chain, this
        // works.
        if (root->loop_op.getNumResults() > 0 && new_loop &&
            new_loop.getNumResults() == root->loop_op.getNumResults()) {
          root->loop_op.replaceAllUsesWith(new_loop.getResults());
        }
      }

      // Erases the original root loop.
      root->loop_op.erase();
    }

    return success();
  }

  void printSALT(SALTNode *node, int depth) {
    for (int i = 0; i < depth; ++i) {
      llvm::errs() << "  ";
    }
    llvm::errs() << "Loop [" << node->lower_bound << "," << node->upper_bound
                 << ") step=" << node->step
                 << " | body_ops=" << node->body_operations.size()
                 << " | children=" << node->children.size() << "\n";
    for (SALTNode *child : node->children) {
      printSALT(child, depth + 1);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createAffineLoopTreeSerializationPass() {
  return std::make_unique<AffineLoopTreeSerializationPass>();
}