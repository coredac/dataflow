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

void printDFGStatistics(mlir::neura::DFGGraph* graph) {
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
  
  llvm::DenseSet<Operation*> patternOps(instance.operations.begin(), instance.operations.end());
  
  Operation* lastOp = instance.lastOp;
  for (Operation* op : instance.operations) {
    for (Value result : op->getResults()) {
      for (OpOperand& use : result.getUses()) {
        Operation* user = use.getOwner();
        if (!patternOps.contains(user) && user->getBlock() == lastOp->getBlock() && user->isBeforeInBlock(lastOp)) {
          return false;
        }
      }
    }
  }
  
  builder.setInsertionPointAfter(lastOp);
  
  llvm::SetVector<Value> inputSet;
  for (Operation* op : instance.operations) {
    for (Value operand : op->getOperands()) {
      Operation* defOp = operand.getDefiningOp();
      if (defOp && defOp->getName().getStringRef().str() == "neura.fused_op" && patternOps.contains(defOp)) {
        continue;
      }
      if (!defOp || !patternOps.contains(defOp)) {
        inputSet.insert(operand);
      }
    }
    
    if (op->getName().getStringRef().str() == "neura.fused_op" && op->getNumRegions() > 0) {
      Region& region = op->getRegion(0);
      if (!region.empty()) {
        Block& block = region.front();
        llvm::DenseSet<Operation*> nestedPatternOps;
        
        for (Operation& bodyOp : block.getOperations()) {
          if (bodyOp.getName().getStringRef().str() != "neura.yield") {
            nestedPatternOps.insert(&bodyOp);
            for (Value operand : bodyOp.getOperands()) {
              if (mlir::isa<BlockArgument>(operand)) {
                continue;
              }
              
              Operation* defOp = operand.getDefiningOp();
              if (defOp && !nestedPatternOps.contains(defOp) && !patternOps.contains(defOp)) {
                inputSet.insert(operand);
              } else if (!defOp) {
                assert(false && "Value without defining op should not happen normally");
              }
            }
          }
        }
      }
    }
  }
  SmallVector<Value> validInputs = inputSet.takeVector();
  
  llvm::SetVector<Value> outputSet;
  for (Operation* op : instance.operations) {
    for (Value result : op->getResults()) {
      bool hasExternalUse = false;
      for (OpOperand& use : result.getUses()) {
        Operation* user = use.getOwner();
        if (!patternOps.contains(user)) {
          hasExternalUse = true;
          break;
        }
      }
      
      if (hasExternalUse) {
        outputSet.insert(result);
      }
    }
  }
  SmallVector<Value> validOutputs = outputSet.takeVector();
  
  SmallVector<Type> outputTypes;
  for (Value output : validOutputs) {
    outputTypes.push_back(output.getType());
  }
  
  llvm::errs() << "    Creating fused_op operation: " << pattern.getPattern() << " "<< "Inputs: " << validInputs.size() << " " << "Outputs: " << outputTypes.size() << "\n";

  auto patternOp = builder.create<neura::FusedOpOp>(
      lastOp->getLoc(),
      outputTypes,
      validInputs,
      builder.getI64IntegerAttr(pattern.getId()),
      builder.getStringAttr(pattern.getPattern()),
      builder.getI64IntegerAttr(pattern.getFrequency())
  );

  Region& bodyRegion = patternOp.getBody();
  Block* bodyBlock = new Block();
  bodyRegion.push_back(bodyBlock);
  
  for (Value input : validInputs) {
    bodyBlock->addArgument(input.getType(), input.getLoc());
  }
  
  builder.setInsertionPointToStart(bodyBlock);
  IRMapping mapping;
  
  for (size_t i = 0; i < validInputs.size(); ++i) {
    mapping.map(validInputs[i], bodyBlock->getArgument(i));
  }
  
  llvm::DenseMap<Value, Value> originalToCloned;
  
  Operation* clonedOp = nullptr;

  for (Operation* op : instance.operations) {
    if (op->getName().getStringRef().str() == "neura.fused_op") {
      if (op->getNumRegions() > 0) {
        Region& region = op->getRegion(0);
        if (!region.empty()) {
          Block& block = region.front();
          
          llvm::DenseSet<Operation*> nestedPatternBodyOps;
          llvm::SetVector<Value> nestedPatternUsedValues;
          
          for (Operation& bodyOp : block.getOperations()) {
            if (bodyOp.getName().getStringRef().str() != "neura.yield") {
              nestedPatternBodyOps.insert(&bodyOp);
              
              for (Value operand : bodyOp.getOperands()) {
                if (mlir::isa<BlockArgument>(operand)) {
                  continue;
                }
                
                Operation* defOp = operand.getDefiningOp();
                if (defOp) {
                  if (nestedPatternBodyOps.contains(defOp)) {
                    continue;
                  }
                  if (!patternOps.contains(defOp)) {
                    nestedPatternUsedValues.insert(operand);
                  }
                }
              }
            }
          }
          
          for (size_t i = 0; i < op->getNumOperands() && i < block.getNumArguments(); ++i) {
            Value patternInput = op->getOperand(i);
            BlockArgument nestedArg = block.getArgument(i);
            
            if (mapping.contains(patternInput)) {
              mapping.map(nestedArg, mapping.lookup(patternInput));
            } else {
              if (originalToCloned.count(patternInput)) {
                mapping.map(nestedArg, originalToCloned[patternInput]);
              } else {
                mapping.map(nestedArg, patternInput);
              }
            }
          }
          
          for (Value usedVal : nestedPatternUsedValues) {
            if (mlir::isa<BlockArgument>(usedVal) || mapping.contains(usedVal)) {
              continue;
            }
            
            Operation* defOp = usedVal.getDefiningOp();
            if (defOp && patternOps.contains(defOp) && originalToCloned.count(usedVal)) {
              mapping.map(usedVal, originalToCloned[usedVal]);
            } else {
              mapping.map(usedVal, usedVal);
            }
          }
          
          for (Operation& bodyOp : block.getOperations()) {
            if (bodyOp.getName().getStringRef().str() != "neura.yield") {
              clonedOp = builder.clone(bodyOp, mapping);
              for (size_t i = 0; i < bodyOp.getNumResults(); ++i) {
                originalToCloned[bodyOp.getResult(i)] = clonedOp->getResult(i);
              }
            }
          }
          
          for (Operation& blockOp : block.getOperations()) {
            if (blockOp.getName().getStringRef().str() == "neura.yield") {
              for (size_t i = 0; i < op->getNumResults() && i < blockOp.getNumOperands(); ++i) {
                Value yieldOperand = blockOp.getOperand(i);
                if (originalToCloned.count(yieldOperand)) {
                  originalToCloned[op->getResult(i)] = originalToCloned[yieldOperand];
                  mapping.map(op->getResult(i), originalToCloned[yieldOperand]);
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
        Operation* defOp = operand.getDefiningOp();
        if (defOp && defOp->getName().getStringRef().str() == "neura.fused_op" && 
            patternOps.contains(defOp) && originalToCloned.count(operand)) {
          if (!mapping.contains(operand)) {
            mapping.map(operand, originalToCloned[operand]);
          }
        }
      }
      clonedOp = builder.clone(*op, mapping);
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        originalToCloned[op->getResult(i)] = clonedOp->getResult(i);
      }
    }
  }
  
  SmallVector<Value> yieldOperands;
  for (size_t i = 0; i < validOutputs.size(); ++i) {
    Value originalOutput = validOutputs[i];
    if (originalToCloned.count(originalOutput)) {
      Value clonedValue = originalToCloned[originalOutput];
      if (clonedValue) {
        yieldOperands.push_back(clonedValue);
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  
  builder.create<neura::YieldOp>(lastOp->getLoc(), yieldOperands);
  
  llvm::DenseSet<Value> replacedOutputs;
  for (size_t i = 0; i < validOutputs.size(); ++i) {
    Value oldValue = validOutputs[i];
    Value newValue = patternOp.getResult(i);
    oldValue.replaceAllUsesWith(newValue);
    replacedOutputs.insert(oldValue);
  }
  
  for (auto& pair : originalToCloned) {
    Value oldValue = pair.first;
    if (replacedOutputs.contains(oldValue)) {
      continue;
    }
    if (!oldValue.use_empty()) {
      Value newValue = pair.second;
      oldValue.replaceAllUsesWith(newValue);
    }
  }
  
  originalToCloned.clear();
  
  for (auto it = instance.operations.rbegin(); it != instance.operations.rend(); ++it) {
    Operation* op = *it;
  
    if (op->getName().getStringRef().str() == "neura.fused_op") {
      Region& region = op->getRegion(0);
      Block& block = region.front();
      
      for (Operation& bodyOp : block.getOperations()) {
        for (Value result : bodyOp.getResults()) {
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
        Operation& bodyOp = block.back();
        bodyOp.dropAllReferences();
        bodyOp.erase();
      }
    }
    
    op->dropAllUses();
    op->erase();
  }
  
  return true;
}

int rewritePatternsToRegions(mlir::neura::DFGGraph* dfg_graph, ModuleOp module_op, const std::vector<mlir::neura::PatternWithSelectedInstances>& patterns_with_instances) {
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

  Option<int> minSupport{
      *this, "min-support",
      llvm::cl::desc("Minimum support threshold for pattern mining (default: 1)"),
      llvm::cl::init(1)};
  Option<int> maxIter{
      *this, "max-iter",
      llvm::cl::desc("Maximum number of iterations for pattern merging (default: 2)"),
      llvm::cl::init(2)};

  void runOnOperation() override {
    
    ModuleOp module_op = getOperation();
    
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "IterMergePatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << minSupport.getValue() << "\n";
    llvm::errs() << "========================================\n\n";
    int iter = 0;
    while (iter < maxIter.getValue()) {
      llvm::errs() << "Iteration " << iter << "\n";
      auto dfg_graph = mlir::neura::DFGExtractor::extractFromModule(module_op);
      
      if (!dfg_graph) {
        llvm::errs() << "Error: Failed to extract DFG from module\n";
        signalPassFailure();
        return;
      } 
      
      printDFGStatistics(dfg_graph.get());
      mlir::neura::GraMi grami(dfg_graph.get(), minSupport.getValue());
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

  Option<int> minSupport{
      *this, "min-support",
      llvm::cl::desc("Minimum support threshold for pattern mining (default: 1)"),
      llvm::cl::init(1)};

  void runOnOperation() override {
    ModuleOp module_op = getOperation();
    
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "InitPatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << minSupport.getValue() << "\n";
    llvm::errs() << "========================================\n\n";
    
    auto dfg_graph = mlir::neura::DFGExtractor::extractFromModule(module_op);
    
    if (!dfg_graph) {
      llvm::errs() << "Error: Failed to extract DFG from module\n";
      signalPassFailure();
      return;
    } 
    
    printDFGStatistics(dfg_graph.get());
    mlir::neura::GraMi grami(dfg_graph.get(), minSupport.getValue());
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
