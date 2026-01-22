#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/Transforms/GraphMining/GraMi.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace mlir;

#define GEN_PASS_DEF_ITERMERGEPATTERN
#define GEN_PASS_DEF_INITPATTERN
#include "NeuraDialect/NeuraPasses.h.inc"

void printDFGStatistics(mlir::neura::DfgGraph *graph) {
  llvm::errs() << "DFG Statistics:\n";
  llvm::errs() << "---------------\n";

  llvm::errs() << "Number of nodes: " << graph->getNumNodes() << "\n";
  llvm::errs() << "Number of edges: " << graph->getNumEdges() << "\n\n";

  std::map<std::string, size_t> op_type_counts;
  for (auto *node : graph->getNodes()) {
    op_type_counts[node->getLabel()]++;
  }

  llvm::errs() << "Operation types and their counts:\n";
  for (const auto &pair : op_type_counts) {
    llvm::errs() << "  - " << pair.first << ": " << pair.second << "\n";
  }
  llvm::errs() << "\n";
}

// Finds a valid insertion point for the fused operation.
Operation *
findValidInsertionPoint(const mlir::neura::PatternInstance &instance,
                        const llvm::DenseSet<Operation *> &pattern_ops,
                        const SmallVector<Value> &valid_inputs,
                        const SmallVector<Value> &valid_outputs) {

  if (instance.operations.empty())
    return nullptr;

  Block *block = instance.operations.front()->getBlock();
  if (!block)
    return nullptr;

  for (Operation *op : instance.operations) {
    if (op->getBlock() != block) {
      return nullptr;
    }
  }

  Operation *earliest_point = nullptr;

  for (Value input : valid_inputs) {
    Operation *def_op = input.getDefiningOp();
    if (!def_op) {
      continue;
    }

    if (def_op->getBlock() != block) {
      continue;
    }

    if (!earliest_point) {
      earliest_point = def_op;
    } else if (!def_op->isBeforeInBlock(earliest_point)) {
      earliest_point = def_op;
    }
  }

  // Finds the latest position: before all external uses of outputs
  Operation *latest_point = nullptr;

  for (Value output : valid_outputs) {
    for (OpOperand &use : output.getUses()) {
      Operation *user = use.getOwner();

      if (pattern_ops.contains(user)) {
        continue;
      }

      if (user->getBlock() != block) {
        continue;
      }

      if (!latest_point) {
        latest_point = user;
      } else if (user->isBeforeInBlock(latest_point)) {
        latest_point = user;
      }
    }
  }

  if (!earliest_point) {
    earliest_point = instance.operations.front();
    for (Operation *op : instance.operations) {
      if (op->isBeforeInBlock(earliest_point)) {
        earliest_point = op;
      }
    }
  }

  // [earliest_point, latest_point)
  if (latest_point) {
    if (!earliest_point->isBeforeInBlock(latest_point) ||
        earliest_point == latest_point) {
      return nullptr;
    }
  }

  // Returns the valid insertion point (inserts after earliest_point)
  return earliest_point;
}

bool rewritePatternInstance(OpBuilder &builder,
                            const mlir::neura::PatternInstance &instance,
                            const mlir::neura::FrequentSubgraph &pattern) {
  if (instance.operations.empty())
    return false;

  for (Operation *op : instance.operations) {
    if (!op || !op->getBlock()) {
      return false;
    }
  }

  llvm::DenseSet<Operation *> pattern_ops(instance.operations.begin(),
                                          instance.operations.end());

  // First, collects inputs and outputs to determine valid insertion point
  llvm::SetVector<Value> input_set_for_check;
  for (Operation *op : instance.operations) {
    for (Value operand : op->getOperands()) {
      Operation *def_op = operand.getDefiningOp();
      if (def_op &&
          def_op->getName().getStringRef().str() == "neura.fused_op" &&
          pattern_ops.contains(def_op)) {
        continue;
      }
      if (!def_op || !pattern_ops.contains(def_op)) {
        input_set_for_check.insert(operand);
      }
    }

    if (op->getName().getStringRef().str() == "neura.fused_op" &&
        op->getNumRegions() > 0) {
      Region &region = op->getRegion(0);
      if (!region.empty()) {
        Block &block = region.front();
        llvm::DenseSet<Operation *> nested_pattern_ops;

        for (Operation &body_op : block.getOperations()) {
          if (body_op.getName().getStringRef().str() != "neura.yield") {
            nested_pattern_ops.insert(&body_op);
            for (Value operand : body_op.getOperands()) {
              if (mlir::isa<BlockArgument>(operand)) {
                continue;
              }

              Operation *def_op = operand.getDefiningOp();
              if (def_op && !nested_pattern_ops.contains(def_op) &&
                  !pattern_ops.contains(def_op)) {
                input_set_for_check.insert(operand);
              } else if (!def_op) {
                assert(false &&
                       "Value without defining op should not happen normally");
              }
            }
          }
        }
      }
    }
  }
  SmallVector<Value> valid_inputs = input_set_for_check.takeVector();

  llvm::SetVector<Value> output_set_for_check;
  for (Operation *op : instance.operations) {
    for (Value result : op->getResults()) {
      bool has_external_use = false;
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (!pattern_ops.contains(user)) {
          has_external_use = true;
          break;
        }
      }

      if (has_external_use) {
        output_set_for_check.insert(result);
      }
    }
  }
  SmallVector<Value> valid_outputs = output_set_for_check.takeVector();

  // Finds a valid insertion point that avoids dominance issues
  Operation *insertion_point = findValidInsertionPoint(
      instance, pattern_ops, valid_inputs, valid_outputs);
  if (!insertion_point) {
    return false;
  }

  builder.setInsertionPointAfter(insertion_point);

  SmallVector<Type> output_types;
  for (Value output : valid_outputs) {
    output_types.push_back(output.getType());
  }

  auto pattern_op = builder.create<neura::FusedOp>(
      insertion_point->getLoc(), output_types, valid_inputs,
      builder.getI64IntegerAttr(pattern.getId()),
      builder.getStringAttr(pattern.getPattern()),
      builder.getI64IntegerAttr(pattern.getFrequency()));

  Region &body_region = pattern_op.getBody();
  Block *body_block = new Block();
  body_region.push_back(body_block);

  for (Value input : valid_inputs) {
    body_block->addArgument(input.getType(), input.getLoc());
  }

  builder.setInsertionPointToStart(body_block);
  IRMapping mapping;

  for (size_t i = 0; i < valid_inputs.size(); ++i) {
    mapping.map(valid_inputs[i], body_block->getArgument(i));
  }

  llvm::DenseMap<Value, Value> original_to_cloned;

  Operation *cloned_op = nullptr;

  for (Operation *op : instance.operations) {
    if (op->getName().getStringRef().str() == "neura.fused_op") {
      if (op->getNumRegions() > 0) {
        Region &region = op->getRegion(0);
        if (!region.empty()) {
          Block &block = region.front();

          llvm::DenseSet<Operation *> nested_pattern_body_ops;
          llvm::SetVector<Value> nested_pattern_used_values;

          for (Operation &body_op : block.getOperations()) {
            if (body_op.getName().getStringRef().str() != "neura.yield") {
              nested_pattern_body_ops.insert(&body_op);

              for (Value operand : body_op.getOperands()) {
                if (mlir::isa<BlockArgument>(operand)) {
                  continue;
                }

                Operation *def_op = operand.getDefiningOp();
                if (def_op) {
                  if (nested_pattern_body_ops.contains(def_op)) {
                    continue;
                  }
                  if (!pattern_ops.contains(def_op)) {
                    nested_pattern_used_values.insert(operand);
                  }
                }
              }
            }
          }

          for (size_t i = 0;
               i < op->getNumOperands() && i < block.getNumArguments(); ++i) {
            Value pattern_input = op->getOperand(i);
            BlockArgument nested_arg = block.getArgument(i);

            if (mapping.contains(pattern_input)) {
              mapping.map(nested_arg, mapping.lookup(pattern_input));
            } else {
              if (original_to_cloned.count(pattern_input)) {
                mapping.map(nested_arg, original_to_cloned[pattern_input]);
              } else {
                mapping.map(nested_arg, pattern_input);
              }
            }
          }

          for (Value used_val : nested_pattern_used_values) {
            if (mlir::isa<BlockArgument>(used_val) ||
                mapping.contains(used_val)) {
              continue;
            }

            Operation *def_op = used_val.getDefiningOp();
            if (def_op && pattern_ops.contains(def_op) &&
                original_to_cloned.count(used_val)) {
              mapping.map(used_val, original_to_cloned[used_val]);
            } else {
              mapping.map(used_val, used_val);
            }
          }

          for (Operation &body_op : block.getOperations()) {
            if (body_op.getName().getStringRef().str() != "neura.yield") {
              cloned_op = builder.clone(body_op, mapping);
              for (size_t i = 0; i < body_op.getNumResults(); ++i) {
                original_to_cloned[body_op.getResult(i)] =
                    cloned_op->getResult(i);
              }
            }
          }

          for (Operation &block_op : block.getOperations()) {
            if (block_op.getName().getStringRef().str() == "neura.yield") {
              for (size_t i = 0;
                   i < op->getNumResults() && i < block_op.getNumOperands();
                   ++i) {
                Value yield_operand = block_op.getOperand(i);
                if (original_to_cloned.count(yield_operand)) {
                  original_to_cloned[op->getResult(i)] =
                      original_to_cloned[yield_operand];
                  mapping.map(op->getResult(i),
                              original_to_cloned[yield_operand]);
                } else {
                  return false;
                }
              }
              break;
            }
          }
        }
      }
    } else {
      for (Value operand : op->getOperands()) {
        Operation *def_op = operand.getDefiningOp();
        if (def_op &&
            def_op->getName().getStringRef().str() == "neura.fused_op" &&
            pattern_ops.contains(def_op) && original_to_cloned.count(operand)) {
          if (!mapping.contains(operand)) {
            mapping.map(operand, original_to_cloned[operand]);
          }
        }
      }
      cloned_op = builder.clone(*op, mapping);
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        original_to_cloned[op->getResult(i)] = cloned_op->getResult(i);
      }
    }
  }

  SmallVector<Value> yield_operands;
  for (size_t i = 0; i < valid_outputs.size(); ++i) {
    Value original_output = valid_outputs[i];
    if (original_to_cloned.count(original_output)) {
      Value cloned_value = original_to_cloned[original_output];
      if (cloned_value) {
        yield_operands.push_back(cloned_value);
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  builder.create<neura::YieldOp>(insertion_point->getLoc(), ValueRange{},
                                 yield_operands);

  llvm::DenseSet<Value> replaced_outputs;
  for (size_t i = 0; i < valid_outputs.size(); ++i) {
    Value old_value = valid_outputs[i];
    Value new_value = pattern_op.getResult(i);
    old_value.replaceAllUsesWith(new_value);
    replaced_outputs.insert(old_value);
  }

  for (auto &pair : original_to_cloned) {
    Value old_value = pair.first;
    if (replaced_outputs.contains(old_value)) {
      continue;
    }
    if (!old_value.use_empty()) {
      Value new_value = pair.second;
      old_value.replaceAllUsesWith(new_value);
    }
  }

  original_to_cloned.clear();

  for (auto it = instance.operations.rbegin(); it != instance.operations.rend();
       ++it) {
    Operation *op = *it;

    if (op->getName().getStringRef().str() == "neura.fused_op") {
      Region &region = op->getRegion(0);
      Block &block = region.front();

      for (Operation &body_op : block.getOperations()) {
        for (Value result : body_op.getResults()) {
          if (!result.use_empty()) {
            result.dropAllUses();
          }
        }
      }

      for (BlockArgument arg : block.getArguments()) {
        if (!arg.use_empty()) {
          arg.dropAllUses();
        }
      }

      while (!block.empty()) {
        Operation &body_op = block.back();
        body_op.dropAllReferences();
        body_op.erase();
      }
    }

    op->dropAllUses();
    op->erase();
  }

  return true;
}

int rewritePatternsToRegions(
    mlir::neura::DfgGraph *dfg_graph, ModuleOp module_op,
    const std::vector<mlir::neura::PatternWithSelectedInstances>
        &patterns_with_instances) {
  int rewrite_count = 0;
  size_t total_critical = 0;
  size_t total_non_critical = 0;
  MLIRContext *context = module_op.getContext();
  OpBuilder builder(context);

  for (const auto &pwsi : patterns_with_instances) {
    if (pwsi.pattern.getNodes().size() < 2)
      continue;
    total_critical += pwsi.critical_instances.size();
    total_non_critical += pwsi.non_critical_instances.size();
  }

  size_t total_instances = total_critical + total_non_critical;
  if (total_instances == 0) {
    llvm::errs() << "  No valid instances to rewrite\n";
    return 0;
  }

  std::set<std::string> attempted_patterns;

  // Phase 1: Rewrites all critical path instances across all patterns
  llvm::errs() << "  Phase 1: Rewriting critical path instances...\n";
  for (const auto &pwsi : patterns_with_instances) {
    if (pwsi.pattern.getNodes().size() < 2 || pwsi.critical_instances.empty()) {
      continue;
    }

    attempted_patterns.insert(pwsi.pattern.getPattern());

    for (const auto &instance : pwsi.critical_instances) {
      rewritePatternInstance(builder, instance, pwsi.pattern);
    }
  }

  // Phase 2: Rewrites all non-critical path instances across all patterns
  llvm::errs() << "  Phase 2: Rewriting non-critical path instances...\n";
  for (const auto &pwsi : patterns_with_instances) {
    if (pwsi.pattern.getNodes().size() < 2 ||
        pwsi.non_critical_instances.empty()) {
      continue;
    }

    // Marks pattern as attempted before trying to fuse instances
    attempted_patterns.insert(pwsi.pattern.getPattern());

    for (const auto &instance : pwsi.non_critical_instances) {
      rewritePatternInstance(builder, instance, pwsi.pattern);
    }
  }

  // Marks all attempted patterns
  for (const auto &pattern_str : attempted_patterns) {
    mlir::neura::GraMi::markPatternAsAttempted(pattern_str);
  }

  return rewrite_count;
}

namespace {

struct IterMergePatternPass
    : public PassWrapper<IterMergePatternPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IterMergePatternPass)

  IterMergePatternPass() = default;
  IterMergePatternPass(const IterMergePatternPass &pass)
      : PassWrapper<IterMergePatternPass, OperationPass<ModuleOp>>(pass) {}

  StringRef getArgument() const override { return "iter-merge-pattern"; }
  StringRef getDescription() const override {
    return "Iteratively merge and identify common patterns in DFG using graph "
           "mining.";
  }

  Option<int> min_support{
      *this, "min-support",
      llvm::cl::desc(
          "Minimum support threshold for pattern mining (default: 2)"),
      llvm::cl::init(2)};
  Option<int> max_iter{
      *this, "max-iter",
      llvm::cl::desc(
          "Maximum number of iterations for pattern merging (default: 2)"),
      llvm::cl::init(2)};

  void runOnOperation() override {

    ModuleOp module_op = getOperation();

    llvm::errs() << "\n========================================\n";
    llvm::errs() << "IterMergePatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << min_support.getValue()
                 << "\n";
    llvm::errs() << "========================================\n\n";

    int iter = 0;
    bool cleared_attempted =
        false; // Tracks if it has cleared attempted marks once
    while (iter < max_iter.getValue()) {
      llvm::errs() << "Iteration " << iter << "\n";

      // Re-collects critical path operations from all functions for this
      // iteration Critical path may change after each iteration due to pattern
      // fusion
      llvm::DenseSet<Operation *> all_critical_ops;
      module_op.walk([&](func::FuncOp func) {
        auto critical_ops = mlir::neura::GraMi::collectCriticalPathOps(func);
        for (Operation *op : critical_ops) {
          all_critical_ops.insert(op);
        }
      });
      llvm::errs() << "  Collected " << all_critical_ops.size()
                   << " critical path operations for iteration " << iter
                   << "\n";

      auto dfg_graph = mlir::neura::DfgExtractor::extractFromModule(module_op);

      if (!dfg_graph) {
        llvm::errs() << "Error: Failed to extract DFG from module\n";
        signalPassFailure();
        return;
      }

      printDFGStatistics(dfg_graph.get());
      mlir::neura::GraMi grami(dfg_graph.get(), min_support.getValue());
      grami.setCriticalPathOps(all_critical_ops);
      std::vector<mlir::neura::PatternWithSelectedInstances>
          patterns_with_instances = grami.mineFrequentSubgraphs();

      // If no patterns were fused and it hasn't cleared attempted marks yet,
      // clears them and tries one more iteration (without incrementing iter
      // count)
      if (patterns_with_instances.empty() && !cleared_attempted) {
        llvm::errs() << "  No patterns fused in this iteration. Clearing "
                        "attempted marks and retrying...\n";
        mlir::neura::GraMi::clearAttemptedPatterns();
        cleared_attempted = true;
        // Retries this iteration with cleared marks (doesn't increment iter)
        continue;
      }

      // If it cleared marks and still got 0, or if it has reached max
      // iterations, stops
      if (patterns_with_instances.empty() && cleared_attempted) {
        llvm::errs() << "  No patterns fused even after clearing attempted "
                        "marks. Stopping.\n";
        break;
      }

      int rewrite_count = rewritePatternsToRegions(dfg_graph.get(), module_op,
                                                   patterns_with_instances);
      llvm::errs() << "  - Rewrote " << rewrite_count
                   << " pattern instances\n\n";

      iter++;
    }

    llvm::errs() << "\n========================================\n";
    llvm::errs() << "IterMergePatternPass: Completed\n";
    llvm::errs() << "========================================\n\n";
  }
};

struct InitPatternPass
    : public PassWrapper<InitPatternPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitPatternPass)

  InitPatternPass() = default;
  InitPatternPass(const InitPatternPass &pass)
      : PassWrapper<InitPatternPass, OperationPass<ModuleOp>>(pass) {}

  StringRef getArgument() const override { return "init-pattern"; }
  StringRef getDescription() const override {
    return "Initialize and identify common patterns in DFG (single iteration).";
  }

  Option<int> min_support{
      *this, "min-support",
      llvm::cl::desc(
          "Minimum support threshold for pattern mining (default: 2)"),
      llvm::cl::init(2)};

  void runOnOperation() override {
    ModuleOp module_op = getOperation();

    llvm::errs() << "\n========================================\n";
    llvm::errs() << "InitPatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << min_support.getValue()
                 << "\n";
    llvm::errs() << "========================================\n\n";

    // Collects critical path operations from all functions
    llvm::DenseSet<Operation *> all_critical_ops;
    module_op.walk([&](func::FuncOp func) {
      auto critical_ops = mlir::neura::GraMi::collectCriticalPathOps(func);
      for (Operation *op : critical_ops) {
        all_critical_ops.insert(op);
      }
    });
    llvm::errs() << "Collected " << all_critical_ops.size()
                 << " critical path operations\n\n";

    auto dfg_graph = mlir::neura::DfgExtractor::extractFromModule(module_op);

    if (!dfg_graph) {
      llvm::errs() << "Error: Failed to extract DFG from module\n";
      signalPassFailure();
      return;
    }

    printDFGStatistics(dfg_graph.get());
    mlir::neura::GraMi grami(dfg_graph.get(), min_support.getValue());
    grami.setCriticalPathOps(all_critical_ops);
    std::vector<mlir::neura::PatternWithSelectedInstances>
        patterns_with_instances = grami.mineFrequentSubgraphs();

    int rewrite_count = rewritePatternsToRegions(dfg_graph.get(), module_op,
                                                 patterns_with_instances);
    llvm::errs() << "  - Rewrote " << rewrite_count << " pattern instances\n\n";

    llvm::errs() << "\n========================================\n";
    llvm::errs() << "InitPatternPass: Completed\n";
    llvm::errs() << "========================================\n\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createIterMergePatternPass() {
  return std::make_unique<IterMergePatternPass>();
}

std::unique_ptr<Pass> createInitPatternPass() {
  return std::make_unique<InitPatternPass>();
}
} // namespace mlir::neura
