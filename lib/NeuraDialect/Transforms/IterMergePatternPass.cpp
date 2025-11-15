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
    
    llvm::outs() << "\n========================================\n";
    llvm::outs() << "IterMergePatternPass: Starting pattern mining\n";
    llvm::outs() << "Minimum support threshold: " << minSupport << "\n";
    llvm::outs() << "========================================\n\n";
    
    // Step 2.5: Merge adjacent patterns
    int iter = 0;
    while (iter < 2) {
      // Step 1: Extract DFG from the module
      llvm::outs() << "Iteration " << iter << "\n";
      llvm::outs() << "Step 1: Extracting DFG from module...\n";
      auto dfg_graph = mlir::neura::DFGExtractor::extractFromModule(module_op);
      
      if (!dfg_graph) {
        llvm::errs() << "Error: Failed to extract DFG from module\n";
        signalPassFailure();
        return;
      } 
      
      // Prints the statistics of the DFG
      printDFGStatistics(dfg_graph.get());

      // Step 2: Mine frequent subgraphs using GraMi
      llvm::outs() << "Step 2: Mining frequent subgraphs...\n";
      mlir::neura::GraMi grami(dfg_graph.get(), minSupport);
      std::vector<mlir::neura::PatternWithSelectedInstances> patterns_with_instances = 
          grami.mineFrequentSubgraphs();
      
      llvm::outs() << "  - Found " << patterns_with_instances.size() << " patterns with selected instances\n\n";
      // patterns_with_instances = grami.mergeAdjacentPatterns(patterns_with_instances);
      llvm::outs() << "  - After merging: " << patterns_with_instances.size() << " patterns remaining\n\n";
      
      // Step 3: Rewrite operations to wrap patterns in regions
      int rewrite_count = rewritePatternsToRegions(module_op, patterns_with_instances);
      llvm::outs() << "  - Rewrote " << rewrite_count << " pattern instances\n\n";
      iter++;
    }
    
    llvm::outs() << "\n========================================\n";
    llvm::outs() << "IterMergePatternPass: Completed\n";
    llvm::outs() << "========================================\n\n";
  }
  
private:
  
  void printDFGStatistics(mlir::neura::DFGGraph* graph) {
    llvm::outs() << "DFG Statistics:\n";
    llvm::outs() << "---------------\n";

    llvm::outs() << "Number of nodes: " << graph->getNumNodes() << "\n";
    llvm::outs() << "Number of edges: " << graph->getNumEdges() << "\n\n";
    
    // Counts s  operations by type
    std::map<std::string, size_t> op_type_counts;
    for (auto* node : graph->getNodes()) {
      op_type_counts[node->getLabel()]++;
    }
    
    llvm::outs() << "Operation types and their counts:\n";
    for (const auto& pair : op_type_counts) {
      llvm::outs() << "  - " << pair.first << ": " << pair.second << "\n";
    }
    llvm::outs() << "\n";
  }
  
  void printCommonPatterns(const std::vector<CommonPatternInfo>& patterns) {
    llvm::outs() << "========================================\n";
    llvm::outs() << "Common Patterns Found:\n";
    llvm::outs() << "========================================\n\n";
    
    if (patterns.empty()) {
      llvm::outs() << "No common patterns found with the given threshold.\n";
      return;
    }
    
    // Sort patterns by frequency (descending) and size
    std::vector<CommonPatternInfo> sorted_patterns = patterns;
    std::sort(sorted_patterns.begin(), sorted_patterns.end(),
              [](const CommonPatternInfo& a, const CommonPatternInfo& b) {
                if (a.frequency != b.frequency) {
                  return a.frequency > b.frequency;
                }
                return a.nodes.size() > b.nodes.size();
              });
    
    int pattern_id = 1;
    for (const auto& pattern : sorted_patterns) {
      llvm::outs() << "Pattern #" << pattern_id << ":\n";
      llvm::outs() << "  Pattern String: " << pattern.pattern_string << "\n";
      llvm::outs() << "  Frequency: " << pattern.frequency << "\n";
      llvm::outs() << "  Size: " << pattern.nodes.size() << " nodes, " 
                   << pattern.edges.size() << " edges\n";
      
      // Print nodes
      if (!pattern.nodes.empty()) {
        llvm::outs() << "  Nodes:\n";
        for (const auto& node_pair : pattern.nodes) {
          llvm::outs() << "    Node " << node_pair.first << ": " 
                       << node_pair.second << "\n";
        }
      }
      
      // Print edges
      if (!pattern.edges.empty()) {
        llvm::outs() << "  Edges:\n";
        for (const auto& edge_pair : pattern.edges) {
          llvm::outs() << "    Edge " << edge_pair.first << ": " 
                       << edge_pair.second.first << " -> " 
                       << edge_pair.second.second << "\n";
        }
      }
      
      llvm::outs() << "\n";
      pattern_id++;
    }
  }
  
  void storeCommonPatternsAsAttribute(ModuleOp module_op, 
                                      const std::vector<CommonPatternInfo>& patterns) {
    // Store pattern information as module attributes for use by other passes
    MLIRContext* context = module_op.getContext();
    
    // Store the number of patterns found
    module_op->setAttr("neura.num_common_patterns", 
                       IntegerAttr::get(IntegerType::get(context, 64), 
                                       patterns.size()));
    
    // Store summary information
    if (!patterns.empty()) {
      // Find the most frequent pattern
      auto max_freq_pattern = std::max_element(
          patterns.begin(), patterns.end(),
          [](const CommonPatternInfo& a, const CommonPatternInfo& b) {
            return a.frequency < b.frequency;
          });
      
      module_op->setAttr("neura.max_pattern_frequency",
                        IntegerAttr::get(IntegerType::get(context, 64),
                                        max_freq_pattern->frequency));
      
      llvm::outs() << "Stored pattern information in module attributes:\n";
      llvm::outs() << "  - neura.num_common_patterns = " << patterns.size() << "\n";
      llvm::outs() << "  - neura.max_pattern_frequency = " 
                   << max_freq_pattern->frequency << "\n\n";
    }
  }
  
  // Rewrite patterns by wrapping them in regions
  int rewritePatternsToRegions(ModuleOp module_op, 
                                const std::vector<mlir::neura::PatternWithSelectedInstances>& patterns_with_instances) {
    int rewrite_count = 0;
    
    // Extract DFG once for efficiency
    auto dfg_graph = mlir::neura::DFGExtractor::extractFromModule(module_op);
    if (!dfg_graph) {
      llvm::errs() << "Error: Failed to extract DFG for pattern analysis\n";
      return 0;
    }
    
    // Filter valid patterns (at least 2 nodes) and collect all instances
    std::vector<std::tuple<PatternInstance, const mlir::neura::FrequentSubgraph*, int>> all_instances;
    int pattern_id = 0;
    
    for (const auto& pwsi : patterns_with_instances) {
      if (pwsi.pattern.getNodes().size() >= 2 && !pwsi.selected_instances.empty()) {
        llvm::outs() << "  Valid pattern: " << pwsi.pattern.getPattern() 
                     << " with " << pwsi.selected_instances.size() << " instances\n";
        
        // Print pattern nodes with their DFG IDs
        llvm::outs() << "    Pattern nodes:\n";
        for (const auto& node : pwsi.pattern.getNodes()) {
          llvm::outs() << "      Pattern node " << node.first << ": " << node.second << "\n";
        }
        
        // Print actual DFG node IDs from the first instance
        if (!pwsi.selected_instances.empty()) {
          llvm::outs() << "    DFG node IDs from first instance:\n";
          const auto& first_instance = pwsi.selected_instances[0];
          for (size_t i = 0; i < first_instance.operations.size(); ++i) {
            mlir::Operation* op = first_instance.operations[i];
            // Find the corresponding DFG node
            for (auto* dfg_node : dfg_graph->getNodes()) {
              if (dfg_node->getOperation() == op) {
                llvm::outs() << "      Instance node " << i << ": " << op->getName() 
                             << " (DFG ID: " << dfg_node->getId() << ")\n";
                break;
              }
            }
          }
        }
        // for (const auto& edge : pwsi.pattern.getEdges()) {
        //   llvm::outs() << "    Edge: " << edge.first << " -> " << edge.second.first << " -> " << edge.second.second << "\n";
        // }

        for (const auto& instance : pwsi.selected_instances) {
          all_instances.push_back({instance, &pwsi.pattern, pattern_id});
        }
      }
      pattern_id++;
    }
    
    if (all_instances.empty()) {
      llvm::outs() << "  No valid instances to rewrite\n";
      return 0;
    }
    
    llvm::outs() << "  Total instances to rewrite: " << all_instances.size() << "\n";
    
    MLIRContext* context = module_op.getContext();
    OpBuilder builder(context);
    
    // Process each instance
    int instance_num = 0;
    for (auto& [instance, pattern, pid] : all_instances) {
      instance_num++;
      llvm::outs() << "    Instance " << instance_num << "/" << all_instances.size() 
                   << " (Pattern #" << pid << " - " << pattern->getPattern() << ")...\n";
      
      if (rewritePatternInstance(builder, instance, *pattern, pid)) {
        rewrite_count++;
        llvm::outs() << "      Success!\n";
      } else {
        llvm::outs() << "      Skipped (operation already deleted).\n";
      }
    }
    
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
                               const mlir::neura::FrequentSubgraph& pattern,
                               int pattern_id) {
    if (instance.operations.empty()) return false;
    
    // Check if any operation in this instance has already been erased
    for (Operation* op : instance.operations) {
      if (!op || !op->getBlock()) {
        llvm::outs() << "    Skipping instance: operation already erased\n";
        return false;
      }
    }
    
    // Build a set of operations in this pattern for quick lookup
    llvm::DenseSet<Operation*> patternOps(instance.operations.begin(), instance.operations.end());

    llvm::outs() << "    Pattern operations: " << patternOps.size() << "\n";
    for (Operation* op : patternOps) {
      llvm::outs() << "      " << op->getName() << "\n";
      if (op->getName().getStringRef().str() == "neura.common_pattern") {
        // get the body operations
        if (op->getNumRegions() > 0) {
          for (Operation& bodyOp : op->getRegion(0).getOps()) {
            llvm::outs() << "        " << bodyOp.getName() << "\n";
          }
        }
      }
    }
    
    // // Check for domination safety: ensure no value produced by the pattern
    // // is used by operations that come BEFORE the pattern's lastOp
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
            llvm::outs() << "    Skipping instance: would violate domination\n";
            llvm::outs() << "      Value from " << op->getName() 
                         << " is used by " << user->getName() 
                         << " which comes before the pattern\n";
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
        // Only include operands from outside the pattern
        Operation* defOp = operand.getDefiningOp();
        if (!defOp || !patternOps.contains(defOp)) {
          inputSet.insert(operand);
        }
      }
    }
    SmallVector<Value> validInputs = inputSet.takeVector();
    
    // Recalculate outputs dynamically
    llvm::SetVector<Value> outputSet;
    for (Operation* op : instance.operations) {
      for (Value result : op->getResults()) {
        // Check if this result is used outside the pattern
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
    
    llvm::outs() << "    Creating common_pattern operation: " << pattern.getPattern() << "\n";
    llvm::outs() << "      Inputs: " << validInputs.size() << "\n";
    llvm::outs() << "      Outputs: " << outputTypes.size() << "\n";

    auto patternOp = builder.create<neura::CommonPatternOp>(
        lastOp->getLoc(),
        outputTypes,
        validInputs,
        builder.getI64IntegerAttr(pattern_id),
        builder.getStringAttr(pattern.getPattern()),
        builder.getI64IntegerAttr(pattern.getFrequency())
    );
    
    llvm::outs() << "    Created common_pattern operation successfully\n";

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
      llvm::outs() << "    Cloning operation: " << op->getName() << "\n";
      if (op->getName().getStringRef().str() == "neura.common_pattern") {
        // For pattern operations, we need to handle their inputs and outputs specially
        if (op->getNumRegions() > 0) {
          Region& region = op->getRegion(0);
          if (!region.empty()) {
            Block& block = region.front();
            
            // Handle BlockArguments: map pattern inputs correctly
            for (size_t i = 0; i < op->getNumOperands(); ++i) {
              Value patternInput = op->getOperand(i);
              
              // Check if this input is an external input (has corresponding block argument)
              bool isExternalInput = false;
              for (size_t j = 0; j < validInputs.size(); ++j) {
                if (validInputs[j] == patternInput) {
                  // This is an external input, map to block argument
                  BlockArgument newArg = bodyBlock->getArgument(j);
                  if (i < block.getNumArguments()) {
                    mapping.map(block.getArgument(i), newArg);
                    llvm::outs() << "      Mapped external input " << i << " to block argument " << j << "\n";
                  }
                  isExternalInput = true;
                  break;
                }
              }
              
              if (!isExternalInput) {
                // This is an internal input (from another pattern in the same instance)
                // Map directly to the source value
                if (i < block.getNumArguments()) {
                  mapping.map(block.getArgument(i), patternInput);
                  llvm::outs() << "      Mapped internal input " << i << " to source value\n";
                }
              }
            }
            
            // Clone all operations in the pattern's body
            for (Operation& bodyOp : block.getOperations()) {
              llvm::outs() << "      Cloning body operation: " << bodyOp.getName() << "\n";
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
                    llvm::outs() << "      Mapped pattern result " << i << " to cloned value\n";
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
    llvm::outs() << "    Building yield operands for " << validOutputs.size() << " outputs\n";
    for (size_t i = 0; i < validOutputs.size(); ++i) {
      Value originalOutput = validOutputs[i];
      if (originalToCloned.count(originalOutput)) {
        Value clonedValue = originalToCloned[originalOutput];
        if (clonedValue) {
          yieldOperands.push_back(clonedValue);
          llvm::outs() << "      Output " << i << ": mapped to valid cloned value\n";
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
    // Step 1: Replace external output values with pattern op results
    for (size_t i = 0; i < validOutputs.size(); ++i) {
      Value oldValue = validOutputs[i];
      Value newValue = patternOp.getResult(i);
      oldValue.replaceAllUsesWith(newValue);
      llvm::outs() << "      Replaced output " << i << " with pattern op result\n";
    }
    
    // Step 2: Replace all cloned values with their new counterparts
    for (auto& pair : originalToCloned) {
      Value oldValue = pair.first;
      Value newValue = pair.second;
      oldValue.replaceAllUsesWith(newValue);
    }
    
    // Step 3: Handle pattern operations specially
    for (Operation* op : instance.operations) {
      if (op->getName().getStringRef().str() == "neura.common_pattern") {
        // Replace pattern operation results with new pattern operation results
        for (size_t i = 0; i < op->getNumResults() && i < patternOp.getNumResults(); ++i) {
          Value oldValue = op->getResult(i);
          Value newValue = patternOp.getResult(i);
          llvm::outs() << "      Replacing use of " << oldValue.getDefiningOp()->getName() << " with " << newValue.getDefiningOp()->getName() << "\n";
          oldValue.replaceAllUsesWith(newValue);
        }
      }
    }
    
    // Step 4: Final cleanup - replace any remaining uses with themselves to avoid assertion
    for (Operation* op : instance.operations) {
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          llvm::outs() << "      Replacing use of " << result << " with itself\n";
          // Force replace with itself to avoid assertion failure
          result.replaceAllUsesWith(result);
        }
      }
    }
    
    // Clear the mapping to avoid dangling references
    originalToCloned.clear();

    // print the new pattern op
    llvm::outs() << "    New pattern op: " << patternOp.getPatternId() << "\n";
    llvm::outs() << "      Inputs: " << patternOp.getInputs().size() << "\n";
    llvm::outs() << "      Outputs: " << patternOp.getOutputs().size() << "\n";
    
    for (Operation& op : patternOp.getBody().getOps()) {
      llvm::outs() << "        Body operation: " << op.getName() << " (ID: " << op.getAttr("id") << ")\n";
    }

    // Erase original operations by first replacing all uses, then deleting
    for (auto it = instance.operations.rbegin(); it != instance.operations.rend(); ++it) {
      Operation* op = *it;
      
      // if (op->getName().getStringRef().str() == "neura.common_pattern") {
      //   llvm::outs() << "    Skipping common pattern operation: " << op->getName() << "\n";
      //   continue;
      // }
      
      llvm::outs() << "    Erasing operation: " << op->getName() << "\n";
      
      // List all uses of the operation before deletion
      for (Operation* user : op->getUsers()) {
        llvm::outs() << "        User: " << user->getName() << "\n";
      }

      if (op->use_empty()) {
        llvm::outs() << "        Operation is empty, skipping deletion\n";
      }
      else {
        llvm::outs() << "        Operation is not empty, dropping uses\n";
      }
      
      // Special handling for common_pattern operations
      if (op->getName().getStringRef().str() == "neura.common_pattern") {
        Region& region = op->getRegion(0);
        Block& block = region.front();
        
        // Step 0: Handle BlockArguments that are used within the region
        llvm::outs() << "        Checking " << block.getNumArguments() << " block arguments\n";
        for (BlockArgument arg : block.getArguments()) {
          if (!arg.use_empty()) {
            // Count uses manually
            int useCount = 0;
            for (OpOperand& use : arg.getUses()) {
              useCount++;
            }
            llvm::outs() << "        Dropping uses of block argument " << arg.getArgNumber() 
                         << " (type: " << arg.getType() << ", uses: " << useCount << ")\n";
            // Print all users of this argument
            for (OpOperand& use : arg.getUses()) {
              llvm::outs() << "          Used by: " << use.getOwner()->getName() << " (ID: " << use.getOwner() << ")\n";
            }
            arg.dropAllUses();
            llvm::outs() << "        Block argument " << arg.getArgNumber() << " uses dropped\n";
          } else {
            llvm::outs() << "        Block argument " << arg.getArgNumber() 
                         << " (type: " << arg.getType() << ") has no uses, skipping\n";
          }
        }
        
        // Step 1: Drop all references from region operations first
        for (Operation& bodyOp : block.getOperations()) {
          bodyOp.dropAllReferences();
        }
        
        // Step 2: Clear all uses of region operations' results
        for (Operation& bodyOp : block.getOperations()) {
          for (Value result : bodyOp.getResults()) {
            if (!result.use_empty()) {
              llvm::outs() << "        Dropping uses of region result: " << result << "\n";
              result.dropAllUses();
            }
          }
        }
        
        // Step 3: Now safely erase all operations in reverse order
        while (!block.empty()) {
          Operation& bodyOp = block.back();
          llvm::outs() << "        Erasing region operation: " << bodyOp.getName() << "\n";
          bodyOp.erase();
        }
        llvm::outs() << "        Region operations erased\n";
      }
      
      op->dropAllUses();
      
      // Replace all uses of this operation's results with dummy values
      // This breaks the dependency chain and allows safe deletion
      
      // Now it's safe to erase the operation
      op->erase();
      llvm::outs() << "    Operation erased: " << op->getName() << "\n";
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
