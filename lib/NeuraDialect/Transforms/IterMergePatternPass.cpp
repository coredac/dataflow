#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/GraphMining/GraMi.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <vector>
#include <map>
#include <set>

using namespace mlir;

#define GEN_PASS_DEF_ITERMERGEPATTERN
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

// Data structure to store common patterns
struct CommonPatternInfo {
  std::string pattern_string;
  size_t frequency;
  std::map<size_t, std::string> nodes;
  std::map<size_t, std::pair<size_t, size_t>> edges;
  
  CommonPatternInfo(const std::string& pattern, size_t freq)
    : pattern_string(pattern), frequency(freq) {}
};

struct IterMergePatternPass
    : public PassWrapper<IterMergePatternPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IterMergePatternPass)

  StringRef getArgument() const override { return "iter-merge-pattern"; }
  StringRef getDescription() const override {
    return "Iteratively merge and identify common patterns in DFG using graph mining.";
  }

  void runOnOperation() override {
    // Default minimum support threshold
    // TODO: Make this a command line argument
    int minSupport = 1;
    
    ModuleOp module_op = getOperation();
    
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "IterMergePatternPass: Starting pattern mining\n";
    llvm::errs() << "Minimum support threshold: " << minSupport << "\n";
    llvm::errs() << "========================================\n\n";
    
    // Step 2.5: Merge adjacent patterns
    // TODO: Make this a command line argument
    int maxIterations = 2;
    int iter = 0;
    while (iter < maxIterations) {
      // Step 1: Extract DFG from the module
      llvm::errs() << "Iteration " << iter << "\n";
      llvm::errs() << "Step 1: Extracting DFG from module...\n";
      auto dfg_graph = mlir::neura::DFGExtractor::extractFromModule(module_op);
      
      if (!dfg_graph) {
        llvm::errs() << "Error: Failed to extract DFG from module\n";
        signalPassFailure();
        return;
      } 
      
      // Prints the statistics of the DFG
      printDFGStatistics(dfg_graph.get());

      // Step 2: Mine frequent subgraphs using GraMi
      llvm::errs() << "Step 2: Mining frequent subgraphs...\n";
      mlir::neura::GraMi grami(dfg_graph.get(), minSupport);
      std::vector<mlir::neura::PatternWithSelectedInstances> patterns_with_instances = 
          grami.mineFrequentSubgraphs();
      
      llvm::errs() << "  - Found " << patterns_with_instances.size() << " patterns with selected instances\n\n";
      
      // Step 3: Rewrite operations to wrap patterns in regions
      int rewrite_count = rewritePatternsToRegions(dfg_graph.get(), module_op, patterns_with_instances);
      llvm::errs() << "  - Rewrote " << rewrite_count << " pattern instances\n\n";
      iter++;
    }
    
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "IterMergePatternPass: Completed\n";
    llvm::errs() << "========================================\n\n";
  }
  
private:
  
  void printDFGStatistics(mlir::neura::DFGGraph* graph) {
    llvm::errs() << "DFG Statistics:\n";
    llvm::errs() << "---------------\n";

    llvm::errs() << "Number of nodes: " << graph->getNumNodes() << "\n";
    llvm::errs() << "Number of edges: " << graph->getNumEdges() << "\n\n";
    
    // Counts s  operations by type
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
  
  // Rewrite patterns by wrapping them in regions
  int rewritePatternsToRegions(mlir::neura::DFGGraph* dfg_graph, ModuleOp module_op, 
                                const std::vector<mlir::neura::PatternWithSelectedInstances>& patterns_with_instances) {
    int rewrite_count = 0;
    
    // Filter and process valid patterns (at least 2 nodes) directly without accumulating
    size_t total_instances = 0;
    MLIRContext* context = module_op.getContext();
    OpBuilder builder(context);
    
    // Process each instance
    int instance_num = 0;
    for (const auto& pwsi : patterns_with_instances) {
      if (pwsi.pattern.getNodes().size() < 2 || pwsi.selected_instances.empty()) continue;
      for (const auto& instance : pwsi.selected_instances) {
        total_instances++;
        instance_num++;
        const auto* pattern = &pwsi.pattern;
        llvm::errs() << "    Instance " << instance_num << " (Pattern #" << pattern->getId() << " - " << pattern->getPattern() << ")...\n";
        if (rewritePatternInstance(builder, instance, *pattern)) {
          rewrite_count++;
          llvm::errs() << "      Success!\n";
        } else {
          llvm::errs() << "      Skipped (operation already deleted).\n";
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
  
  // Use the PatternInstance from GraMi header
  using PatternInstance = mlir::neura::PatternInstance;
  
  // Get operation label for pattern matching (matches DFGExtractor logic)
  std::string getOperationPatternLabel(Operation* op) {
    std::string opName = op->getName().getStringRef().str();
    
    // Remove dialect prefixs
    size_t dotPos = opName.find('.');
    if (dotPos != std::string::npos) {
      opName = opName.substr(dotPos + 1);
    }
    
    // Add type information
    if (op->getNumResults() > 0) {
      Type resultType = op->getResult(0).getType();
      if (auto intType = mlir::dyn_cast<IntegerType>(resultType)) {
        opName += "_i" + std::to_string(intType.getWidth());
      } else if (auto floatType = mlir::dyn_cast<FloatType>(resultType)) {
        opName += "_f" + std::to_string(floatType.getWidth());
      }
    }
    
    return opName;
  }
  
  // Rewrite a single pattern instance
  bool rewritePatternInstance(OpBuilder& builder, const PatternInstance& instance,
                               const mlir::neura::FrequentSubgraph& pattern) {
    if (instance.operations.empty()) return false;
    
    // Check if any operation in this instance has already been erased
    for (Operation* op : instance.operations) {
      if (!op || !op->getBlock()) {
        llvm::errs() << "    Skipping instance: operation already erased\n";
        return false;
      }
    }
    
    // Build a set of operations in this pattern for quick lookup
    llvm::DenseSet<Operation*> patternOps(instance.operations.begin(), instance.operations.end());
    
    // Check for domination safety: ensure no value produced by the pattern is used by operations that come BEFORE the pattern's lastOp
    Operation* lastOp = instance.lastOp;
    for (Operation* op : instance.operations) {
      for (Value result : op->getResults()) {
        for (OpOperand& use : result.getUses()) {
          Operation* user = use.getOwner();
          // If an external user comes before lastOp, we have a domination issue
          // Only check domination if both operations are in the same block
          if (!patternOps.contains(user) && 
              user->getBlock() == lastOp->getBlock() && 
              user->isBeforeInBlock(lastOp)) {
            llvm::errs() << "    Skipping instance: would violate domination\n";
            llvm::errs() << "      Value from " << op->getName() << " is used by " << user->getName() << " which comes before the pattern\n";
            return false;
          }
        }
      }
    }
    
    // Set insertion point after the last operation in the pattern
    builder.setInsertionPointAfter(lastOp);
    
    // Recalculate inputs dynamically (to get current values after previous rewrites)
    llvm::SetVector<Value> inputSet;
    for (Operation* op : instance.operations) {
      for (Value operand : op->getOperands()) {
        // Only include operands from errside the pattern
        Operation* defOp = operand.getDefiningOp();
        if (!defOp || !patternOps.contains(defOp)) {
          inputSet.insert(operand);
        }
      }
      
      // For nested pattern operations, also collect values used inside their bodies
      // But exclude values that are produced within the nested pattern itself
      if (op->getName().getStringRef().str() == "neura.common_pattern" && op->getNumRegions() > 0) {
        Region& region = op->getRegion(0);
        if (!region.empty()) {
          Block& block = region.front();
          
          // First, collect all operations defined within this nested pattern
          llvm::DenseSet<Operation*> nestedPatternOps;
          for (Operation& bodyOp : block.getOperations()) {
            if (bodyOp.getName().getStringRef().str() != "neura.yield") {
              nestedPatternOps.insert(&bodyOp);
            }
          }
          
          // Then, collect values used in the nested pattern that come from errside
          for (Operation& bodyOp : block.getOperations()) {
            if (bodyOp.getName().getStringRef().str() != "neura.yield") {
              for (Value operand : bodyOp.getOperands()) {
                // Skip block arguments (these are the nested pattern's own inputs)
                if (mlir::isa<BlockArgument>(operand)) {
                  continue;
                }
                
                Operation* defOp = operand.getDefiningOp();
                
                // Only include values that are:
                // 1. Defined errside the current pattern instance (not in patternOps)
                // 2. AND not defined within this nested pattern's body (not in nestedPatternOps)
                if (defOp) {
                  // If defined within the nested pattern itself, skip it
                  if (nestedPatternOps.contains(defOp)) {
                    continue;
                  }
                  // If defined errside the pattern instance, add it as input
                  if (!patternOps.contains(defOp)) {
                    inputSet.insert(operand);
                  }
                } else {
                  // Value without defining op (shouldn't happen normally)
                  // But if it exists and is not a block argument, it might be needed
                  // Skip for now to be safe
                  assert(false && "Value without defining op should not happen normally");
                }
              }
            }
          }
        }
      }
    }
    SmallVector<Value> validInputs = inputSet.takeVector();
    
    // Recalculate outputs dynamically
    llvm::SetVector<Value> outputSet;
    for (Operation* op : instance.operations) {
      for (Value result : op->getResults()) {
        // Check if this result is used errside the pattern
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
    
    // Create the common_pattern operation
    SmallVector<Type> outputTypes;
    for (Value output : validOutputs) {
      outputTypes.push_back(output.getType());
    }
    
    llvm::errs() << "    Creating common_pattern operation: " << pattern.getPattern() << "\n";
    llvm::errs() << "      Inputs: " << validInputs.size() << "\n";
    llvm::errs() << "      Outputs: " << outputTypes.size() << "\n";

    auto patternOp = builder.create<neura::CommonPatternOp>(
        lastOp->getLoc(),
        outputTypes,
        validInputs,
        builder.getI64IntegerAttr(pattern.getId()),
        builder.getStringAttr(pattern.getPattern()),
        builder.getI64IntegerAttr(pattern.getFrequency())
    );
    
    llvm::errs() << "    Created common_pattern operation successfully\n";

    // Create the body region
    Region& bodyRegion = patternOp.getBody();
    Block* bodyBlock = new Block();
    bodyRegion.push_back(bodyBlock);
    
    // Add block arguments for valid inputs
    for (Value input : validInputs) {
      bodyBlock->addArgument(input.getType(), input.getLoc());
    }
    
    // Clone operations into the region
    builder.setInsertionPointToStart(bodyBlock);
    IRMapping mapping;
    
    // Map external inputs to block arguments
    // Always map inputs to block arguments to ensure proper dominance
    for (size_t i = 0; i < validInputs.size(); ++i) {
      mapping.map(validInputs[i], bodyBlock->getArgument(i));
    }
    
    // Clone operations in order and build mapping from original to cloned values
    llvm::DenseMap<Value, Value> originalToCloned;
    
    Operation* clonedOp = nullptr;

    for (Operation* op : instance.operations) {
      llvm::errs() << "    Cloning operation: " << op->getName() << "\n";
      if (op->getName().getStringRef().str() == "neura.common_pattern") {
        // For pattern operations, we need to handle their inputs and outputs specially
        if (op->getNumRegions() > 0) {
          Region& region = op->getRegion(0);
          if (!region.empty()) {
            Block& block = region.front();
            
            // Collect all operations defined within this nested pattern and values that need to be mapped
            // Exclude values defined within the nested pattern itself or within the current pattern instance
            llvm::DenseSet<Operation*> nestedPatternBodyOps;
            llvm::SetVector<Value> nestedPatternUsedValues;
            
            for (Operation& bodyOp : block.getOperations()) {
              if (bodyOp.getName().getStringRef().str() != "neura.yield") {
                // Collect the operation
                nestedPatternBodyOps.insert(&bodyOp);
                
                // Collect values used by this operation that need to be mapped
                for (Value operand : bodyOp.getOperands()) {
                  // Skip block arguments (these are the nested pattern's own inputs)
                  if (mlir::isa<BlockArgument>(operand)) {
                    continue;
                  }
                  
                  Operation* defOp = operand.getDefiningOp();
                  
                  // Only collect values that are:
                  // 1. Not defined within the nested pattern's body (not in nestedPatternBodyOps)
                  // 2. Not defined within the current pattern instance (not in patternOps)
                  if (defOp) {
                    // If defined within the nested pattern itself, skip it
                    if (nestedPatternBodyOps.contains(defOp)) {
                      continue;
                    }
                    // If defined outside the pattern instance, add it
                    if (!patternOps.contains(defOp)) {
                      nestedPatternUsedValues.insert(operand);
                    }
                  }
                }
              }
            }
            
            // Handle BlockArguments: map nested pattern's block arguments to outer pattern's block arguments
            // Since validInputs are already mapped to bodyBlock arguments (lines 340-342),
            // we can directly look up the mapping for external inputs
            for (size_t i = 0; i < op->getNumOperands() && i < block.getNumArguments(); ++i) {
              Value patternInput = op->getOperand(i);
              BlockArgument nestedArg = block.getArgument(i);
              
              // Check if this input is an external input (already mapped in lines 340-342)
              if (mapping.contains(patternInput)) {
                // External input: map nested block argument to outer block argument
                mapping.map(nestedArg, mapping.lookup(patternInput));
                llvm::errs() << "      Mapped external input " << i << " to outer block argument\n";
              } else {
                // Internal input: from another pattern in the same instance
                // Try to find the value in cloned operations or use the source value
                Value mappedValue;
                if (originalToCloned.count(patternInput)) {
                  mappedValue = originalToCloned[patternInput];
                } else {
                  // Map to the source value - this should work if the value dominates
                  mappedValue = patternInput;
                  llvm::errs() << "      Mapped internal input " << i << " to source value (may need block arg)\n";
                }
                mapping.map(nestedArg, mappedValue);
              }
            }
            
            // Now handle values used in body that aren't block arguments
            // These might be values from other patterns in the same instance
            // Note: validInputs are already mapped to bodyBlock arguments (lines 340-342)
            for (Value usedVal : nestedPatternUsedValues) {
              if (mlir::isa<BlockArgument>(usedVal) || mapping.contains(usedVal)) {
                continue;
              }
              
              // Check if it's from another pattern in the same instance (should be cloned)
              Operation* defOp = usedVal.getDefiningOp();
              if (defOp && patternOps.contains(defOp) && originalToCloned.count(usedVal)) {
                mapping.map(usedVal, originalToCloned[usedVal]);
                llvm::errs() << "      Mapped nested pattern used value to cloned value\n";
              } else {
                // This value should have been in validInputs (if external) or cloned (if internal)
                llvm::errs() << "      Warning: value used in nested pattern not properly prepared: " << usedVal << "\n";
                // Map to itself as fallback (may fail if not dominated)
                mapping.map(usedVal, usedVal);
              }
            }
            
            // Clone all operations in the pattern's body
            for (Operation& bodyOp : block.getOperations()) {
              llvm::errs() << "      Cloning body operation: " << bodyOp.getName() << "\n";
              if (bodyOp.getName().getStringRef().str() != "neura.yield") {
                clonedOp = builder.clone(bodyOp, mapping);
                // Map the original operation's results to cloned operation's results
                for (size_t i = 0; i < bodyOp.getNumResults(); ++i) {
                  originalToCloned[bodyOp.getResult(i)] = clonedOp->getResult(i);
                }
              }
            }
            
            // Find the yield operation and map pattern outputs to its operands
            for (Operation& blockOp : block.getOperations()) {
              if (blockOp.getName().getStringRef().str() == "neura.yield") {
                // Map pattern op results to yield operands (which are now mapped to cloned values)
                for (size_t i = 0; i < op->getNumResults() && i < blockOp.getNumOperands(); ++i) {
                  Value yieldOperand = blockOp.getOperand(i);
                  if (originalToCloned.count(yieldOperand)) {
                    originalToCloned[op->getResult(i)] = originalToCloned[yieldOperand];
                    llvm::errs() << "      Mapped pattern result " << i << " to cloned value\n";
                  } else {
                    llvm::errs() << "Error: yield operand not found in cloned operations: " << yieldOperand << "\n";
                    return false;
                  }
                }
                break;
              }
            }
          }
        }
        // Note: We don't clone the pattern operation itself, just its body operations
      } else {
        clonedOp = builder.clone(*op, mapping);
        // Map the original operation's results to cloned operation's results
        for (size_t i = 0; i < op->getNumResults(); ++i) {
          originalToCloned[op->getResult(i)] = clonedOp->getResult(i);
        }
      }
    }
    
    // Build yield operands based on validOutputs (in the same order)
    SmallVector<Value> yieldOperands;
    llvm::errs() << "    Building yield operands for " << validOutputs.size() << " outputs\n";
    for (size_t i = 0; i < validOutputs.size(); ++i) {
      Value originalOutput = validOutputs[i];
      if (originalToCloned.count(originalOutput)) {
        Value clonedValue = originalToCloned[originalOutput];
        if (clonedValue) {
          yieldOperands.push_back(clonedValue);
          llvm::errs() << "      Output " << i << ": mapped to valid cloned value\n";
        } else {
          llvm::errs() << "Error: cloned value is null for output " << i << "\n";
          return false;
        }
      } else {
        llvm::errs() << "Error: output value " << i << " not found in cloned operations\n";
        return false;
      }
    }
    
    // Add yield operation
    builder.create<neura::YieldOp>(lastOp->getLoc(), yieldOperands);
    
    // Comprehensive value replacement strategy
    // Replace external output values with pattern op results
    // This handles all outputs including those from nested pattern operations
    llvm::DenseSet<Value> replacedOutputs;
    for (size_t i = 0; i < validOutputs.size(); ++i) {
      Value oldValue = validOutputs[i];
      Value newValue = patternOp.getResult(i);
      oldValue.replaceAllUsesWith(newValue);
      replacedOutputs.insert(oldValue);
      llvm::errs() << "      Replaced output " << i << " with pattern op result\n";
    }
    
    // Replace cloned values that are used externally (not already replaced as outputs)
    // Note: Values used internally within the cloned region are already handled by IRMapping
    for (auto& pair : originalToCloned) {
      Value oldValue = pair.first;
      // Skip if this value was already replaced as an output
      if (replacedOutputs.contains(oldValue)) {
        continue;
      }
      // Only replace if there are external uses (uses outside the pattern instance)
      if (!oldValue.use_empty()) {
        Value newValue = pair.second;
        oldValue.replaceAllUsesWith(newValue);
      }
    }
    
    // Clear the mapping to avoid dangling references
    originalToCloned.clear();

    // print the new pattern op
    llvm::errs() << "    New pattern op: " << patternOp.getPatternId() << "\n";
    llvm::errs() << "      Inputs: " << patternOp.getInputs().size() << "\n";
    llvm::errs() << "      Outputs: " << patternOp.getOutputs().size() << "\n";
    
    for (Operation& op : patternOp.getBody().getOps()) {
      llvm::errs() << "        Body operation: " << op.getName() << " (ID: " << op.getAttr("id") << ")\n";
    }

    // Erase original operations by first replacing all uses, then deleting
    for (auto it = instance.operations.rbegin(); it != instance.operations.rend(); ++it) {
      Operation* op = *it;
      
      llvm::errs() << "    Erasing operation: " << op->getName() << "\n";
      
      // Special handling for common_pattern operations
      if (op->getName().getStringRef().str() == "neura.common_pattern") {
        Region& region = op->getRegion(0);
        Block& block = region.front();
        
        // Handle BlockArguments that are used within the region
        for (BlockArgument arg : block.getArguments()) {
          if (!arg.use_empty()) {
            arg.dropAllUses();
          }
        }
        
        // Drop all references from region operations first
        for (Operation& bodyOp : block.getOperations()) {
          bodyOp.dropAllReferences();
        }
        
        // Clear all uses of region operations' results
        for (Operation& bodyOp : block.getOperations()) {
          for (Value result : bodyOp.getResults()) {
            if (!result.use_empty()) {
              result.dropAllUses();
            }
          }
        }
        
        // Now safely erase all operations in reverse order
        while (!block.empty()) {
          Operation& bodyOp = block.back();
          bodyOp.erase();
        }
      }
      
      op->dropAllUses();
      
      // Now it's safe to erase the operation
      op->erase();
    }
    
    return true;
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createIterMergePatternPass() {
  return std::make_unique<IterMergePatternPass>();
}
} // namespace mlir::neura
