#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/Transforms/GraphMining/GraMi.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>
#include <map>
#include <set>

using namespace mlir;

#define GEN_PASS_DEF_ITERMERGEPATTERN
#define GEN_PASS_DEF_INITPATTERN
#include "NeuraDialect/NeuraPasses.h.inc"

void printDFGStatistics(mlir::neura::DfgGraph* graph) {
  llvm::errs() << "DFG Statistics:\n";
  llvm::errs() << "---------------\n";

  llvm::errs() << "Number of nodes: " << graph->getNumNodes() << "\n";
  llvm::errs() << "Number of edges: " << graph->getNumEdges() << "\n\n";
  
  std::map<std::string, size_t> op_type_counts;
  for (auto* node : graph->getNodes()) {
    op_type_counts[node->getLabel()]++;
  }
  
  llvm::errs() << "Operation types and their counts:\n";
  for (const auto& pair : op_type_counts) {
    llvm::errs() << "  - " << pair.first << ": " << pair.second << "\n";
  }
  llvm::errs() << "\n";
}

bool rewritePatternInstance(OpBuilder& builder, const mlir::neura::PatternInstance& instance, const mlir::neura::FrequentSubgraph& pattern) {
  if (instance.operations.empty()) return false;
  
  for (Operation* op : instance.operations) {
    if (!op || !op->getBlock()) {
      return false;
    }
  }
  
  llvm::DenseSet<Operation*> pattern_ops(instance.operations.begin(), instance.operations.end());
  
  Operation* last_op = instance.last_op;
  for (Operation* op : instance.operations) {
    for (Value result : op->getResults()) {
      for (OpOperand& use : result.getUses()) {
        Operation* user = use.getOwner();
        if (!pattern_ops.contains(user) && user->getBlock() == last_op->getBlock() && user->isBeforeInBlock(last_op)) {
          return false;
        }
      }
    }
  }
  
  builder.setInsertionPointAfter(last_op);
  
  llvm::SetVector<Value> input_set;
  for (Operation* op : instance.operations) {
    for (Value operand : op->getOperands()) {
      Operation* def_op = operand.getDefiningOp();
      if (def_op && def_op->getName().getStringRef().str() == "neura.fused_op" && pattern_ops.contains(def_op)) {
        continue;
      }
      if (!def_op || !pattern_ops.contains(def_op)) {
        input_set.insert(operand);
      }
    }
    
    if (op->getName().getStringRef().str() == "neura.fused_op" && op->getNumRegions() > 0) {
      Region& region = op->getRegion(0);
      if (!region.empty()) {
        Block& block = region.front();
        llvm::DenseSet<Operation*> nested_pattern_ops;
        
        for (Operation& body_op : block.getOperations()) {
          if (body_op.getName().getStringRef().str() != "neura.yield") {
            nested_pattern_ops.insert(&body_op);
            for (Value operand : body_op.getOperands()) {
              if (mlir::isa<BlockArgument>(operand)) {
                continue;
              }
              
              Operation* def_op = operand.getDefiningOp();
              if (def_op && !nested_pattern_ops.contains(def_op) && !pattern_ops.contains(def_op)) {
                input_set.insert(operand);
              } else if (!def_op) {
                assert(false && "Value without defining op should not happen normally");
              }
            }
          }
        }
      }
    }
  }
  SmallVector<Value> valid_inputs = input_set.takeVector();
  
  llvm::SetVector<Value> output_set;
  for (Operation* op : instance.operations) {
    for (Value result : op->getResults()) {
      bool has_external_use = false;
      for (OpOperand& use : result.getUses()) {
        Operation* user = use.getOwner();
        if (!pattern_ops.contains(user)) {
          has_external_use = true;
          break;
        }
      }
      
      if (has_external_use) {
        output_set.insert(result);
      }
    }
  }
  SmallVector<Value> valid_outputs = output_set.takeVector();
  
  SmallVector<Type> output_types;
  for (Value output : valid_outputs) {
    output_types.push_back(output.getType());
  }
  
  llvm::errs() << "    Creating fused_op operation: " << pattern.getPattern() << " "<< "Inputs: " << valid_inputs.size() << " " << "Outputs: " << output_types.size() << "\n";

  auto pattern_op = builder.create<neura::FusedOp>(
      last_op->getLoc(),
      output_types,
      valid_inputs,
      builder.getI64IntegerAttr(pattern.getId()),
      builder.getStringAttr(pattern.getPattern()),
      builder.getI64IntegerAttr(pattern.getFrequency())
  );

  Region& body_region = pattern_op.getBody();
  Block* body_block = new Block();
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
  
  Operation* cloned_op = nullptr;

  for (Operation* op : instance.operations) {
    if (op->getName().getStringRef().str() == "neura.fused_op") {
      if (op->getNumRegions() > 0) {
        Region& region = op->getRegion(0);
        if (!region.empty()) {
          Block& block = region.front();
          
          llvm::DenseSet<Operation*> nested_pattern_body_ops;
          llvm::SetVector<Value> nested_pattern_used_values;
          
          for (Operation& body_op : block.getOperations()) {
            if (body_op.getName().getStringRef().str() != "neura.yield") {
              nested_pattern_body_ops.insert(&body_op);
              
              for (Value operand : body_op.getOperands()) {
                if (mlir::isa<BlockArgument>(operand)) {
                  continue;
                }
                
                Operation* def_op = operand.getDefiningOp();
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
          
          for (size_t i = 0; i < op->getNumOperands() && i < block.getNumArguments(); ++i) {
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
            if (mlir::isa<BlockArgument>(used_val) || mapping.contains(used_val)) {
              continue;
            }
            
            Operation* def_op = used_val.getDefiningOp();
            if (def_op && pattern_ops.contains(def_op) && original_to_cloned.count(used_val)) {
              mapping.map(used_val, original_to_cloned[used_val]);
            } else {
              mapping.map(used_val, used_val);
            }
          }
          
          for (Operation& body_op : block.getOperations()) {
            if (body_op.getName().getStringRef().str() != "neura.yield") {
              cloned_op = builder.clone(body_op, mapping);
              for (size_t i = 0; i < body_op.getNumResults(); ++i) {
                original_to_cloned[body_op.getResult(i)] = cloned_op->getResult(i);
              }
            }
          }
          
          for (Operation& block_op : block.getOperations()) {
            if (block_op.getName().getStringRef().str() == "neura.yield") {
              for (size_t i = 0; i < op->getNumResults() && i < block_op.getNumOperands(); ++i) {
                Value yield_operand = block_op.getOperand(i);
                if (original_to_cloned.count(yield_operand)) {
                  original_to_cloned[op->getResult(i)] = original_to_cloned[yield_operand];
                  mapping.map(op->getResult(i), original_to_cloned[yield_operand]);
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
        Operation* def_op = operand.getDefiningOp();
        if (def_op && def_op->getName().getStringRef().str() == "neura.fused_op" && 
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
  
  builder.create<neura::YieldOp>(last_op->getLoc(), yield_operands);
  
  llvm::DenseSet<Value> replaced_outputs;
  for (size_t i = 0; i < valid_outputs.size(); ++i) {
    Value old_value = valid_outputs[i];
    Value new_value = pattern_op.getResult(i);
    old_value.replaceAllUsesWith(new_value);
    replaced_outputs.insert(old_value);
  }
  
  for (auto& pair : original_to_cloned) {
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
  
  for (auto it = instance.operations.rbegin(); it != instance.operations.rend(); ++it) {
    Operation* op = *it;
  
    if (op->getName().getStringRef().str() == "neura.fused_op") {
      Region& region = op->getRegion(0);
      Block& block = region.front();
      
      for (Operation& body_op : block.getOperations()) {
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
        Operation& body_op = block.back();
        body_op.dropAllReferences();
        body_op.erase();
      }
    }
    
    op->dropAllUses();
    op->erase();
  }
  
  return true;
}

int rewritePatternsToRegions(mlir::neura::DfgGraph* dfg_graph, ModuleOp module_op, const std::vector<mlir::neura::PatternWithSelectedInstances>& patterns_with_instances) {
  int rewrite_count = 0;
  size_t total_instances = 0;
  MLIRContext* context = module_op.getContext();
  OpBuilder builder(context);
  
  for (const auto& pwsi : patterns_with_instances) {
    if (pwsi.pattern.getNodes().size() < 2 || pwsi.selected_instances.empty()) continue;
    for (const auto& instance : pwsi.selected_instances) {
      total_instances++;
      const auto* pattern = &pwsi.pattern;        
      if (rewritePatternInstance(builder, instance, *pattern)) {
        rewrite_count++;
      }
    }
  }
  
  if (total_instances == 0) {
    llvm::errs() << "  No valid instances to rewrite\n";
    return 0;
  }
  
  llvm::errs() << "  Total instances to rewrite: " << total_instances << "\n";
  
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
    return "Iteratively merge and identify common patterns in DFG using graph mining.";
  }

  Option<int> min_support{
      *this, "min-support",
      llvm::cl::desc("Minimum support threshold for pattern mining (default: 2)"),
      llvm::cl::init(2)};
  Option<int> max_iter{
      *this, "max-iter",
      llvm::cl::desc("Maximum number of iterations for pattern merging (default: 2)"),
      llvm::cl::init(2)};

  void runOnOperation() override {
    
    ModuleOp module_op = getOperation();
    
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "IterMergePatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << min_support.getValue() << "\n";
    llvm::errs() << "========================================\n\n";
    int iter = 0;
    while (iter < max_iter.getValue()) {
      llvm::errs() << "Iteration " << iter << "\n";
      auto dfg_graph = mlir::neura::DfgExtractor::extractFromModule(module_op);
      
      if (!dfg_graph) {
        llvm::errs() << "Error: Failed to extract DFG from module\n";
        signalPassFailure();
        return;
      } 
      
      printDFGStatistics(dfg_graph.get());
      mlir::neura::GraMi grami(dfg_graph.get(), min_support.getValue());
      std::vector<mlir::neura::PatternWithSelectedInstances> patterns_with_instances = grami.mineFrequentSubgraphs();
      
      int rewrite_count = rewritePatternsToRegions(dfg_graph.get(), module_op, patterns_with_instances);
      llvm::errs() << "  - Rewrote " << rewrite_count << " pattern instances\n\n";
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
      llvm::cl::desc("Minimum support threshold for pattern mining (default: 2)"),
      llvm::cl::init(2)};

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "InitPatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << min_support.getValue() << "\n";
    llvm::errs() << "========================================\n\n";
    
    auto dfg_graph = mlir::neura::DfgExtractor::extractFromModule(module_op);
    
    if (!dfg_graph) {
      llvm::errs() << "Error: Failed to extract DFG from module\n";
      signalPassFailure();
      return;
    } 
    
    printDFGStatistics(dfg_graph.get());
    mlir::neura::GraMi grami(dfg_graph.get(), min_support.getValue());
    std::vector<mlir::neura::PatternWithSelectedInstances> patterns_with_instances = grami.mineFrequentSubgraphs();
    
    int rewrite_count = rewritePatternsToRegions(dfg_graph.get(), module_op, patterns_with_instances);
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
