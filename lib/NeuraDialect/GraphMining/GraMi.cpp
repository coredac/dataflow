#include "NeuraDialect/GraphMining/GraMi.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace mlir::neura;

// DFGNode implementation
DFGNode* DFGGraph::addNode(mlir::Operation* op, const std::string& label) {
  auto node = new DFGNode(next_node_id_++, op, label);
  nodes_.push_back(node);
  op_to_node_[op] = node;
  return node;
}

DFGEdge* DFGGraph::addEdge(DFGNode* from, DFGNode* to, mlir::Value value) {
  auto edge = new DFGEdge(next_edge_id_++, from, to, value);
  edges_.push_back(edge);
  from->addOutgoingEdge(edge);
  to->addIncomingEdge(edge);
  return edge;
}

DFGNode* DFGGraph::getNode(DFGNode::NodeId id) const {
  if (id < nodes_.size()) {
    return nodes_[id];
  }
  return nullptr;
}

DFGEdge* DFGGraph::getEdge(DFGEdge::EdgeId id) const {
  if (id < edges_.size()) {
    return edges_[id];
  }
  return nullptr;
}

void DFGGraph::clear() {
  for (auto* node : nodes_) {
    delete node;
  }
  for (auto* edge : edges_) {
    delete edge;
  }
  nodes_.clear();
  edges_.clear();
  op_to_node_.clear();
  next_node_id_ = 0;
  next_edge_id_ = 0;
}

// DFGExtractor implementation
std::string DFGExtractor::getOperationLabel(mlir::Operation* op) {
  std::string opName = op->getName().getStringRef().str();
  
  // Remove dialect prefix
  size_t dotPos = opName.find('.');
  if (dotPos != std::string::npos) {
    opName = opName.substr(dotPos + 1);
  }
  
  // Add type information for better pattern recognition
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

bool DFGExtractor::shouldIncludeOperation(mlir::Operation* op) {
  // First, exclude certain operations (check this before dialect check)
  if (op->getName().getStringRef().contains("func.") ||
      op->getName().getStringRef().contains("module") ||
      op->getName().getStringRef().contains("return") || 
      op->getName().getStringRef().contains("data_mov") || 
      op->getName().getStringRef().contains("ctrl_mov") ||
      op->getName().getStringRef().contains("reserve") || 
      op->getName().getStringRef().contains("alloca") ||
      op->getName().getStringRef().contains("yield")) {
    return false;
  }
  
  // Include Neura dialect operations
  if (op->getDialect()->getNamespace() == "neura") {
    return true;
  }
  
  // Include LLVM dialect operations
  if (op->getDialect()->getNamespace() == "llvm") {
    return false;
  }
  
  // Include Arith dialect operations
  if (op->getDialect()->getNamespace() == "arith") {
    return false;
  }
  
  llvm::outs() << "Excluding operation: " << op->getName().getStringRef() << "\n";

  return false;
}

std::unique_ptr<DFGGraph> DFGExtractor::extractFromModule(ModuleOp module) {
  auto graph = std::make_unique<DFGGraph>();
  
  module.walk([&](func::FuncOp func) {
    llvm::outs() << "Extracting DFG from function: " << func.getName() << "\n";

    auto funcGraph = extractFromFunction(func);
    if (funcGraph) {
      // Merge function graphs into the main graph
      for (auto* node : funcGraph->getNodes()) {
        graph->addNode(node->getOperation(), node->getLabel());
      }
      for (auto* edge : funcGraph->getEdges()) {
        graph->addEdge(edge->getFrom(), edge->getTo(), edge->getValue());
      }
    }
  });
  
  return graph;
}

std::unique_ptr<DFGGraph> DFGExtractor::extractFromFunction(func::FuncOp func) {
  auto graph = std::make_unique<DFGGraph>();
  
  // Extract from all blocks in the function
  func.walk([&](Block* block) {
    auto blockGraph = extractFromBlock(block);
    if (blockGraph) {
      // Merge block graphs
      for (auto* node : blockGraph->getNodes()) {
        graph->addNode(node->getOperation(), node->getLabel());
      }
      for (auto* edge : blockGraph->getEdges()) {
        graph->addEdge(edge->getFrom(), edge->getTo(), edge->getValue());
      }
    }
  });
  
  return graph;
}

std::unique_ptr<DFGGraph> DFGExtractor::extractFromBlock(mlir::Block* block) {
  auto graph = std::make_unique<DFGGraph>();
  llvm::DenseMap<mlir::Value, DFGNode*> value_to_node;
  
  // First pass: create nodes for all operations
  for (auto& op : block->getOperations()) {
    if (shouldIncludeOperation(&op)) {
      std::string label = getOperationLabel(&op);
      DFGNode* node = graph->addNode(&op, label);
      
      // Map result values to nodes
      for (mlir::Value result : op.getResults()) {
        value_to_node[result] = node;
      }
    }
  }
  
  // Second pass: create edges based on value usage
  for (auto& op : block->getOperations()) {
    if (shouldIncludeOperation(&op)) {
      DFGNode* currentNode = nullptr;
      
      // Find the current node
      for (mlir::Value result : op.getResults()) {
        if (value_to_node.count(result)) {
          currentNode = value_to_node[result];
          break;
        }
      }
      
      if (!currentNode) continue;
      
      // Create edges from operands to current node
      for (mlir::Value operand : op.getOperands()) {
        if (value_to_node.count(operand)) {
          DFGNode* sourceNode = value_to_node[operand];
          graph->addEdge(sourceNode, currentNode, operand);
        }
      }
    }
  }
  
  return graph;
}

// GraMi algorithm implementation
std::vector<PatternWithSelectedInstances> GraMi::mineFrequentSubgraphs() {
  std::vector<FrequentSubgraph> frequent_subgraphs;

  // Generate 2-node patterns (operation pairs)
  std::map<std::string, size_t> edge_pattern_counts;
  std::map<std::string, std::vector<DFGEdge*>> pattern_edges;
  
  for (auto* edge : graph_->getEdges()) {
    DFGNode* from = edge->getFrom();
    DFGNode* to = edge->getTo();
    std::string pattern = from->getLabel() + "->" + to->getLabel();
    edge_pattern_counts[pattern]++;
    pattern_edges[pattern].push_back(edge);
  }
  
  // Add frequent 2-node patterns
  for (const auto& pair : edge_pattern_counts) {
    if (pair.second >= min_support_) {
      FrequentSubgraph subgraph(pair.first, pair.second);
      subgraph.addNode(0, pair.first.substr(0, pair.first.find("->")));
      subgraph.addNode(1, pair.first.substr(pair.first.find("->") + 2));
      subgraph.addEdge(0, 0, 1);
      frequent_subgraphs.push_back(subgraph);
    }
  }

  // If no frequent patterns found, return empty
  if (frequent_subgraphs.empty()) {
    return {};
  }

  // Collect all instances from all patterns
  std::vector<PatternInstance> all_instances;
  std::vector<size_t> instance_to_pattern;  // Maps instance index to pattern index
  int pattern_id = 0;
  
  for (const auto& subgraph : frequent_subgraphs) {
    size_t pattern_idx = pattern_id++;
    
    // Get all edges that match this pattern
    const auto& edges = pattern_edges[subgraph.getPattern()];
    
    llvm::outs() << "[GraMi] Pattern #" << pattern_idx << " (" << subgraph.getPattern() 
                 << "): " << edges.size() << " total instances\n";
    
    for (auto* edge : edges) {
      PatternInstance instance;
      instance.pattern_id = pattern_idx;
      instance.frequency = 1;
      
      mlir::Operation* fromOp = edge->getFrom()->getOperation();
      mlir::Operation* toOp = edge->getTo()->getOperation();
      
      // Ensure operations are in topological order
      if (fromOp->isBeforeInBlock(toOp)) {
        instance.operations.push_back(fromOp);
        instance.operations.push_back(toOp);
        instance.lastOp = toOp;
      } else {
        instance.operations.push_back(toOp);
        instance.operations.push_back(fromOp);
        instance.lastOp = fromOp;
      }
      
      // Build set of operations for quick lookup
      llvm::DenseSet<mlir::Operation*> patternOps;
      patternOps.insert(fromOp);
      patternOps.insert(toOp);
      
      // Collect inputs: operands from outside the pattern
      llvm::SetVector<mlir::Value> input_set;
      for (mlir::Value operand : fromOp->getOperands()) {
        input_set.insert(operand);
      }
      for (mlir::Value operand : toOp->getOperands()) {
        // Don't include the edge between fromOp and toOp
        if (operand.getDefiningOp() != fromOp) {
          input_set.insert(operand);
        }
      }
      instance.inputs = std::vector<mlir::Value>(input_set.begin(), input_set.end());
      
      // Collect outputs: results used outside the pattern
      llvm::SetVector<mlir::Value> output_set;
      for (mlir::Operation* op : instance.operations) {
        for (mlir::Value result : op->getResults()) {
          // Check if this result is used outside the pattern
          bool hasExternalUse = false;
          for (mlir::OpOperand& use : result.getUses()) {
            mlir::Operation* user = use.getOwner();
            if (!patternOps.contains(user)) {
              hasExternalUse = true;
              break;
            }
          }
          
          if (hasExternalUse) {
            output_set.insert(result);
          }
        }
      }
      instance.outputs = std::vector<mlir::Value>(output_set.begin(), output_set.end());
      
      all_instances.push_back(instance);
      instance_to_pattern.push_back(pattern_idx);
    }
  }
  
  llvm::outs() << "[GraMi] Total instances across all patterns: " << all_instances.size() << "\n";
  
  // Solve maximum weighted independent set on ALL instances at once
  std::vector<size_t> selected_instance_indices = selectMaxWeightedIndependentSetForInstances(all_instances);
  
  llvm::outs() << "[GraMi] Selected " << selected_instance_indices.size() 
               << " non-conflicting instances globally\n";
  
  // Group selected instances by pattern
  std::map<size_t, std::vector<PatternInstance>> pattern_to_instances;
  for (size_t idx : selected_instance_indices) {
    size_t pattern_idx = instance_to_pattern[idx];
    pattern_to_instances[pattern_idx].push_back(all_instances[idx]);
  }
  
  // Build PatternWithInstances with selected instances
  std::vector<PatternWithInstances> valid_patterns;
  std::vector<mlir::neura::FrequentSubgraph> valid_subgraphs;
  
  for (size_t i = 0; i < frequent_subgraphs.size(); ++i) {
    if (pattern_to_instances.count(i) && pattern_to_instances[i].size() >= min_support_) {
      PatternWithInstances pwi;
      pwi.pattern_id = i;
      pwi.instances = pattern_to_instances[i];
      pwi.frequency = pwi.instances.size();
      
      valid_patterns.push_back(pwi);
      valid_subgraphs.push_back(frequent_subgraphs[i]);
      
      llvm::outs() << "[GraMi] Pattern #" << i << " (" << frequent_subgraphs[i].getPattern() 
                   << "): " << pwi.frequency << " non-conflicting instances (meets min_support)\n";
    } else {
      size_t count = pattern_to_instances.count(i) ? pattern_to_instances[i].size() : 0;
      llvm::outs() << "[GraMi] Pattern #" << i << " (" << frequent_subgraphs[i].getPattern() 
                   << "): " << count << " instances < min_support " << min_support_ << " (filtered out)\n";
    }
  }
  
  llvm::outs() << "[GraMi] Found " << valid_patterns.size() 
               << " patterns with sufficient frequency (min_support=" << min_support_ << ")\n";

  if (valid_patterns.empty()) {
    llvm::outs() << "[GraMi] No patterns meet minimum support threshold after instance selection\n";
    return {};
  }

  // Build result with all valid patterns and their selected instances
  // Note: No need for pattern-level MWIS since we already solved global instance-level MWIS
  std::vector<PatternWithSelectedInstances> result;
  
  for (size_t i = 0; i < valid_patterns.size(); ++i) {
    const auto& orig_subgraph = valid_subgraphs[i];
    const auto& pattern_with_inst = valid_patterns[i];
    
    // Create FrequentSubgraph with updated frequency
    FrequentSubgraph updated_subgraph(orig_subgraph.getPattern(), pattern_with_inst.frequency);
    
    // Copy nodes and edges
    for (const auto& node_pair : orig_subgraph.getNodes()) {
      updated_subgraph.addNode(node_pair.first, node_pair.second);
    }
    for (const auto& edge_pair : orig_subgraph.getEdges()) {
      updated_subgraph.addEdge(edge_pair.first, edge_pair.second.first, edge_pair.second.second);
    }
    
    // Create PatternWithSelectedInstances
    PatternWithSelectedInstances pwsi(updated_subgraph);
    pwsi.selected_instances = pattern_with_inst.instances;
    
    result.push_back(pwsi);
  }

  llvm::outs() << "[GraMi] Final result: " << result.size() << " patterns\n";

  // Verify: Check for conflicts between instances (should have none after global MWIS)
  llvm::outs() << "[GraMi] Verifying no conflicts in selected instances...\n";
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      // Check conflicts between instances of different patterns
      for (const auto& inst_i : result[i].selected_instances) {
        for (const auto& inst_j : result[j].selected_instances) {
          if (instancesConflict(inst_i, inst_j)) {
            llvm::errs() << "[GraMi ERROR] Conflict detected between selected instances!\n";
            llvm::errs() << "  Pattern " << i << " and Pattern " << j << "\n";
            llvm::errs() << "  Conflicting operations found\n";
            
            llvm::report_fatal_error(
              "GraMi global MWIS selection produced conflicting instances. "
              "This indicates a bug in the selection algorithm.");
          }
        }
      }
    }
  }
  llvm::outs() << "[GraMi] Verification passed: No conflicts detected.\n";

  return result;
}

std::vector<FrequentSubgraph> GraMi::extendPattern(const FrequentSubgraph& pattern) {
  std::vector<FrequentSubgraph> extensions;
  
  // Find all nodes in the graph that could extend this pattern
  for (auto* node : graph_->getNodes()) {
    std::string nodeLabel = node->getLabel();
    
    // Check if this node type is already in the pattern
    bool nodeTypeExists = false;
    for (const auto& pair : pattern.getNodes()) {
      if (pair.second == nodeLabel) {
        nodeTypeExists = true;
        break;
      }
    }
    
    if (!nodeTypeExists) {
      // Create a new pattern by adding this node
      FrequentSubgraph newPattern = pattern;
      newPattern.addNode(pattern.getNodes().size(), nodeLabel);
      
      // Add edges from existing nodes to the new node
      for (auto* edge : node->getIncomingEdges()) {
        DFGNode* source = edge->getFrom();
        std::string sourceLabel = source->getLabel();
        
        // Find source node in pattern
        for (const auto& pair : pattern.getNodes()) {
          if (pair.second == sourceLabel) {
            newPattern.addEdge(newPattern.getEdges().size(), pair.first, pattern.getNodes().size());
            break;
          }
        }
      }
      
      extensions.push_back(newPattern);
    }
  }
  
  return extensions;
}

bool GraMi::isFrequent(const FrequentSubgraph& candidate) {
  size_t support = countSupport(candidate);
  return support >= min_support_;
}

size_t GraMi::countSupport(const FrequentSubgraph& pattern) {
  
  // Simple support counting based on pattern matching
  // This is a simplified version - in practice, you'd need more sophisticated
  // subgraph isomorphism checking
  
  std::map<std::string, size_t> node_counts;
  for (const auto& pair : pattern.getNodes()) {
    node_counts[pair.second]++;
  }
  
  // Count occurrences of each node type in the graph
  std::map<std::string, size_t> graph_node_counts;
  for (auto* node : graph_->getNodes()) {
    graph_node_counts[node->getLabel()]++;
  }
  
  // Calculate minimum support based on node type frequencies
  size_t min_count = SIZE_MAX;
  for (const auto& pair : node_counts) {
    size_t graph_count = graph_node_counts[pair.first];
    size_t required_count = pair.second;
    if (graph_count < required_count) {
      return 0; // Pattern cannot be frequent
    }
    min_count = std::min(min_count, graph_count / required_count);
  }
  
  return min_count;
}

std::string GraMi::generatePatternString(const FrequentSubgraph& subgraph) {
  std::ostringstream oss;
  oss << "Pattern: ";
  
  // Add nodes
  oss << "Nodes[";
  for (const auto& pair : subgraph.getNodes()) {
    oss << pair.first << ":" << pair.second << " ";
  }
  oss << "] ";
  
  // Add edges
  oss << "Edges[";
  for (const auto& pair : subgraph.getEdges()) {
    oss << pair.first << ":" << pair.second.first << "->" << pair.second.second << " ";
  }
  oss << "] ";
  
  oss << "Support: " << subgraph.getFrequency();
  
  return oss.str();
}

bool GraMi::isIsomorphic(const FrequentSubgraph& pattern1, const FrequentSubgraph& pattern2) {
  // Simple isomorphism check based on node and edge counts
  if (pattern1.getNodes().size() != pattern2.getNodes().size() ||
      pattern1.getEdges().size() != pattern2.getEdges().size()) {
    return false;
  }
  
  // Check if node labels match
  std::map<std::string, size_t> labels1, labels2;
  for (const auto& pair : pattern1.getNodes()) {
    labels1[pair.second]++;
  }
  for (const auto& pair : pattern2.getNodes()) {
    labels2[pair.second]++;
  }
  
  return labels1 == labels2;
}

// Check if two pattern instances conflict (share the same operations/nodes)
bool GraMi::instancesConflict(const PatternInstance& a, const PatternInstance& b) {
  // Two instances conflict if they share any operations (same node)
  for (mlir::Operation* op_a : a.operations) {
    for (mlir::Operation* op_b : b.operations) {
      if (op_a == op_b) return true;
    }
  }
  return false;
}

// Check if two patterns conflict (any of their instances conflict)
bool GraMi::patternsConflict(const PatternWithInstances& a, const PatternWithInstances& b) {
  // Two patterns conflict if ANY of their instances conflict
  for (const auto& inst_a : a.instances) {
    for (const auto& inst_b : b.instances) {
      if (instancesConflict(inst_a, inst_b)) {
        return true;
      }
    }
  }
  return false;
}

// Maximum Weighted Independent Set (MWIS) solver for instances within a single pattern
// Returns indices of selected instances that don't conflict with each other
std::vector<size_t> GraMi::selectMaxWeightedIndependentSetForInstances(
    const std::vector<PatternInstance>& instances) {
  
  if (instances.empty()) return {};
  
  size_t n = instances.size();
  
  // Build conflict graph between instances
  std::vector<std::vector<size_t>> conflicts(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (instancesConflict(instances[i], instances[j])) {
        conflicts[i].push_back(j);
        conflicts[j].push_back(i);
      }
    }
  }
  
  // Greedy algorithm: Select instances by weight/degree ratio
  std::vector<size_t> selected;
  std::vector<bool> available(n, true);
  
  while (true) {
    // Find the best available instance (highest weight / (active_degree + 1) ratio)
    size_t best_idx = n;
    double best_ratio = -1.0;
    
    for (size_t i = 0; i < n; ++i) {
      if (!available[i]) continue;
      
      // Count active degree (neighbors that are still available)
      size_t active_degree = 0;
      for (size_t neighbor : conflicts[i]) {
        if (available[neighbor]) active_degree++;
      }
      
      // All instances have weight 1 for now
      double ratio = 1.0 / (active_degree + 1);
      
      if (ratio > best_ratio) {
        best_ratio = ratio;
        best_idx = i;
      }
    }
    
    if (best_idx == n) break;  // No more available instances
    
    // Select this instance
    selected.push_back(best_idx);
    available[best_idx] = false;
    
    // Mark all conflicting instances as unavailable
    for (size_t neighbor : conflicts[best_idx]) {
      available[neighbor] = false;
    }
  }
  
  return selected;
}

// Maximum Weighted Independent Set (MWIS) solver at pattern level
// Returns indices of selected patterns
std::vector<size_t> GraMi::selectMaxWeightedIndependentSet(
    const std::vector<PatternWithInstances>& patterns) {
  
  if (patterns.empty()) return {};
  
  size_t n = patterns.size();
  
  llvm::outs() << "  [MWIS] Processing " << n << " patterns\n";
  
  // Calculate total instances
  size_t total_instances = 0;
  for (const auto& pattern : patterns) {
    total_instances += pattern.instances.size();
    llvm::outs() << "    Pattern #" << pattern.pattern_id 
                 << ": " << pattern.instances.size() << " instances, "
                 << "frequency=" << pattern.frequency << "\n";
  }
  llvm::outs() << "  [MWIS] Total instances: " << total_instances << "\n";
  
  // Build conflict graph between patterns (adjacency list)
  llvm::outs() << "  [MWIS] Building pattern conflict graph...\n";
  std::vector<std::vector<size_t>> conflicts(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (patternsConflict(patterns[i], patterns[j])) {
        conflicts[i].push_back(j);
        conflicts[j].push_back(i);
      }
    }
  }
  
  // Count total conflicts
  size_t total_conflicts = 0;
  for (const auto& conflict_list : conflicts) {
    total_conflicts += conflict_list.size();
  }
  llvm::outs() << "  [MWIS] Pattern conflict edges: " << total_conflicts / 2 << "\n";
  
  // Greedy algorithm: Select patterns by weight/degree ratio
  std::vector<size_t> selected;
  std::vector<bool> available(n, true);
  
  while (true) {
    // Find the best available pattern (highest weight / (active_degree + 1) ratio)
    size_t best_idx = n;
    double best_ratio = -1.0;
    
    for (size_t i = 0; i < n; ++i) {
      if (!available[i]) continue;
      
      // Count active degree (neighbors that are still available)
      size_t active_degree = 0;
      for (size_t neighbor : conflicts[i]) {
        if (available[neighbor]) active_degree++;
      }
      
      // Use frequency as weight, divide by (active_degree + 1) to prefer:
      // 1. High frequency patterns (better compression)
      // 2. Low degree patterns (less conflicts)
      double ratio = static_cast<double>(patterns[i].frequency) / (active_degree + 1);
      
      if (ratio > best_ratio) {
        best_ratio = ratio;
        best_idx = i;
      }
    }
    
    if (best_idx == n) break;  // No more available patterns
    
    // Select this pattern
    selected.push_back(best_idx);
    available[best_idx] = false;
    
    // Mark all conflicting patterns as unavailable
    for (size_t neighbor : conflicts[best_idx]) {
      available[neighbor] = false;
    }
  }
  
  // Calculate statistics
  size_t total_weight = 0;
  size_t total_selected_instances = 0;
  
  for (size_t idx : selected) {
    total_weight += patterns[idx].frequency;
    total_selected_instances += patterns[idx].instances.size();
  }
  
  llvm::outs() << "  [MWIS] Selected " << selected.size() << "/" << n << " patterns\n";
  llvm::outs() << "  [MWIS] Total instances to rewrite: " << total_selected_instances << "\n";
  llvm::outs() << "  [MWIS] Total weight (sum of frequencies): " << total_weight << "\n";
  llvm::outs() << "  [MWIS] Selected patterns:\n";
  
  for (size_t idx : selected) {
    llvm::outs() << "    Pattern #" << patterns[idx].pattern_id 
                 << " with " << patterns[idx].instances.size() << " instances\n";
  }
  
  return selected;
}

// Merge adjacent common patterns into larger patterns
std::vector<PatternWithSelectedInstances> GraMi::mergeAdjacentPatterns(
    const std::vector<PatternWithSelectedInstances>& patterns) {
  
  if (patterns.size() <= 1) {
    return patterns; // Nothing to merge
  }
  
  llvm::outs() << "\n[MergeAdjacent] Starting pattern merging process...\n";
  llvm::outs() << "[MergeAdjacent] Input patterns: " << patterns.size() << "\n";
  
  std::vector<PatternWithSelectedInstances> final_patterns = {};
  
  for (size_t i = 0; i < patterns.size(); ++i) {
    for (size_t j = i + 1; j < patterns.size(); ++j) {
      const auto& pattern1 = patterns[i];
      const auto& pattern2 = patterns[j];
      
      if (pattern1.pattern.getPattern() == pattern2.pattern.getPattern()) {
        continue;
      }
      
      if (arePatternsAdjacent(pattern1, pattern2)) {
        llvm::outs() << "[MergeAdjacent] Found adjacent patterns: " 
                     << pattern1.pattern.getPattern() << " and " 
                     << pattern2.pattern.getPattern() << "\n";
        
        // Merge the pattern structures using the instance-aware version
        // We'll use the first adjacent instance pair to determine the connection
        PatternInstance* inst1_ptr = nullptr;
        PatternInstance* inst2_ptr = nullptr;
        
        // Find the first adjacent instance pair
        for (const auto& inst1 : pattern1.selected_instances) {
          for (const auto& inst2 : pattern2.selected_instances) {
            if (areInstancesAdjacent(inst1, inst2)) {
              inst1_ptr = const_cast<PatternInstance*>(&inst1);
              inst2_ptr = const_cast<PatternInstance*>(&inst2);
              break;
            }
          }
          if (inst1_ptr) break;
        }
        
        FrequentSubgraph merged_structure("temp", 0);  // Temporary initialization
        if (inst1_ptr && inst2_ptr) {
          // Use the instance-aware version to maintain DFG connections
          merged_structure = mergePatternStructures(*inst1_ptr, *inst2_ptr, pattern1.pattern, pattern2.pattern);
        } else {
          // Fallback to simple merge if no adjacent instances found
          merged_structure = mergePatternStructures(pattern1.pattern, pattern2.pattern);
        }
        
        PatternWithSelectedInstances merged_pattern(merged_structure);
        
        // Merge all adjacent instances between the two patterns
        std::vector<PatternInstance> merged_instances;
        
        for (const auto& inst1 : pattern1.selected_instances) {
          for (const auto& inst2 : pattern2.selected_instances) {
            if (areInstancesAdjacent(inst1, inst2)) {
              PatternInstance merged_instance = mergePatternInstances(inst1, inst2);
              merged_instances.push_back(merged_instance);
            }
          }
        }
        
        merged_pattern.selected_instances = merged_instances;
        
        if (!merged_instances.empty()) {
          llvm::outs() << "[MergeAdjacent] Created merged pattern with " 
                       << merged_instances.size() << " instances\n";
          final_patterns.push_back(merged_pattern);
        }
      }
    }
  }
  
  llvm::outs() << "[MergeAdjacent] Final patterns after merging: " << final_patterns.size() << "\n";
  return final_patterns;
}

// Check if two patterns are adjacent (have adjacent instances)
bool GraMi::arePatternsAdjacent(const PatternWithSelectedInstances& pattern1, 
                               const PatternWithSelectedInstances& pattern2) {
  
  // Two patterns are adjacent if any of their instances are adjacent
  for (const auto& inst1 : pattern1.selected_instances) {
    for (const auto& inst2 : pattern2.selected_instances) {
      if (areInstancesAdjacent(inst1, inst2)) {
        return true;
      }
    }
  }
  
  return false;
}

// Check if two pattern instances are adjacent
bool GraMi::areInstancesAdjacent(const PatternInstance& instance1, 
                                const PatternInstance& instance2) {
  
  // Two instances are adjacent if there's a direct edge between any operation in one instance
  // and any operation in the other instance (treating common_pattern as regular operations)
  
  // Check if any operation in instance1 has a direct edge to any operation in instance2
  for (mlir::Operation* op1 : instance1.operations) {
    for (mlir::Operation* op2 : instance2.operations) {
      // Check if op1 and op2 are in the same block
      if (op1->getBlock() == op2->getBlock()) {
        // Check if there's a direct data flow edge: op1 -> op2
        for (mlir::Value result : op1->getResults()) {
          for (mlir::Value operand : op2->getOperands()) {
            if (result == operand) {
              llvm::outs() << "[MergeAdjacent] Found direct edge: " 
                           << op1->getName() << " -> " << op2->getName() << "\n";
              return true;
            }
          }
        }
        
        // Check if there's a direct data flow edge: op2 -> op1
        for (mlir::Value result : op2->getResults()) {
          for (mlir::Value operand : op1->getOperands()) {
            if (result == operand) {
              llvm::outs() << "[MergeAdjacent] Found direct edge: " 
                           << op2->getName() << " -> " << op1->getName() << "\n";
              return true;
            }
          }
        }
      }
    }
  }
  
  return false;
}

// Merge two pattern structures into one
FrequentSubgraph GraMi::mergePatternStructures(const FrequentSubgraph& pattern1, 
                                              const FrequentSubgraph& pattern2) {
  
  // Create a merged pattern
  std::string merged_pattern_name = pattern1.getPattern() + "+" + pattern2.getPattern();
  size_t merged_frequency = std::min(pattern1.getFrequency(), pattern2.getFrequency());
  
  FrequentSubgraph merged_pattern(merged_pattern_name, merged_frequency);
  
  // Add nodes from both patterns
  size_t node_id = 0;
  std::map<std::string, size_t> label_to_id;
  
  // Add nodes from pattern1
  for (const auto& node_pair : pattern1.getNodes()) {
    if (label_to_id.find(node_pair.second) == label_to_id.end()) {
      label_to_id[node_pair.second] = node_id++;
      merged_pattern.addNode(label_to_id[node_pair.second], node_pair.second);
    }
  }
  
  // Add nodes from pattern2
  for (const auto& node_pair : pattern2.getNodes()) {
    if (label_to_id.find(node_pair.second) == label_to_id.end()) {
      label_to_id[node_pair.second] = node_id++;
      merged_pattern.addNode(label_to_id[node_pair.second], node_pair.second);
    }
  }
  
  // Add edges from both patterns
  size_t edge_id = 0;
  
  // Add edges from pattern1
  for (const auto& edge_pair : pattern1.getEdges()) {
    auto from_label = pattern1.getNodes().at(edge_pair.second.first);
    auto to_label = pattern1.getNodes().at(edge_pair.second.second);
    merged_pattern.addEdge(edge_id++, label_to_id[from_label], label_to_id[to_label]);
  }
  
  // Add edges from pattern2
  for (const auto& edge_pair : pattern2.getEdges()) {
    auto from_label = pattern2.getNodes().at(edge_pair.second.first);
    auto to_label = pattern2.getNodes().at(edge_pair.second.second);
    merged_pattern.addEdge(edge_id++, label_to_id[from_label], label_to_id[to_label]);
  }
  
  return merged_pattern;
}

// Overloaded version that finds the connecting edge between two adjacent instances
FrequentSubgraph GraMi::mergePatternStructures(const PatternInstance& instance1,
                                              const PatternInstance& instance2,
                                              const FrequentSubgraph& pattern1, 
                                              const FrequentSubgraph& pattern2) {
  
  // Create a merged pattern
  std::string merged_pattern_name = pattern1.getPattern() + "+" + pattern2.getPattern();
  size_t merged_frequency = std::min(pattern1.getFrequency(), pattern2.getFrequency());
  
  FrequentSubgraph merged_pattern(merged_pattern_name, merged_frequency);
  
  // Find the connecting edge between the two instances
  mlir::Operation* connecting_from = nullptr;
  mlir::Operation* connecting_to = nullptr;
  std::string connecting_from_label, connecting_to_label;
  
  // Find which operation in instance1 connects to which operation in instance2
  for (mlir::Operation* op1 : instance1.operations) {
    for (mlir::Operation* op2 : instance2.operations) {
      // Check if there's a direct data flow edge: op1 -> op2
      for (mlir::Value result : op1->getResults()) {
        for (mlir::Value operand : op2->getOperands()) {
          if (result == operand) {
            connecting_from = op1;
            connecting_to = op2;
            connecting_from_label = getOperationLabel(op1);
            connecting_to_label = getOperationLabel(op2);
            llvm::outs() << "[MergePatternStructures] Found connecting edge: " 
                         << connecting_from_label << " -> " << connecting_to_label << "\n";
            break;
          }
        }
        if (connecting_from) break;
      }
      if (connecting_from) break;
    }
    if (connecting_from) break;
  }
  
  // If no direct edge found, check reverse direction
  if (!connecting_from) {
    for (mlir::Operation* op2 : instance2.operations) {
      for (mlir::Operation* op1 : instance1.operations) {
        // Check if there's a direct data flow edge: op2 -> op1
        for (mlir::Value result : op2->getResults()) {
          for (mlir::Value operand : op1->getOperands()) {
            if (result == operand) {
              connecting_from = op2;
              connecting_to = op1;
              connecting_from_label = getOperationLabel(op2);
              connecting_to_label = getOperationLabel(op1);
              llvm::outs() << "[MergePatternStructures] Found connecting edge: " 
                           << connecting_from_label << " -> " << connecting_to_label << "\n";
              break;
            }
          }
          if (connecting_from) break;
        }
        if (connecting_from) break;
      }
      if (connecting_from) break;
    }
  }
  
  // Add nodes from both patterns, but maintain the connection structure
  size_t node_id = 0;
  std::map<std::string, size_t> label_to_id;
  
  // Add nodes from pattern1
  for (const auto& node_pair : pattern1.getNodes()) {
    if (label_to_id.find(node_pair.second) == label_to_id.end()) {
      label_to_id[node_pair.second] = node_id++;
      merged_pattern.addNode(label_to_id[node_pair.second], node_pair.second);
    }
  }
  
  // Add nodes from pattern2
  for (const auto& node_pair : pattern2.getNodes()) {
    if (label_to_id.find(node_pair.second) == label_to_id.end()) {
      label_to_id[node_pair.second] = node_id++;
      merged_pattern.addNode(label_to_id[node_pair.second], node_pair.second);
    }
  }
  
  // Add edges from both patterns
  size_t edge_id = 0;
  
  // Add edges from pattern1
  for (const auto& edge_pair : pattern1.getEdges()) {
    auto from_label = pattern1.getNodes().at(edge_pair.second.first);
    auto to_label = pattern1.getNodes().at(edge_pair.second.second);
    merged_pattern.addEdge(edge_id++, label_to_id[from_label], label_to_id[to_label]);
  }
  
  // Add edges from pattern2
  for (const auto& edge_pair : pattern2.getEdges()) {
    auto from_label = pattern2.getNodes().at(edge_pair.second.first);
    auto to_label = pattern2.getNodes().at(edge_pair.second.second);
    merged_pattern.addEdge(edge_id++, label_to_id[from_label], label_to_id[to_label]);
  }
  
  // Add the connecting edge between the two patterns if found
  if (connecting_from && connecting_to && 
      label_to_id.count(connecting_from_label) && label_to_id.count(connecting_to_label)) {
    merged_pattern.addEdge(edge_id++, 
                          label_to_id[connecting_from_label], 
                          label_to_id[connecting_to_label]);
    llvm::outs() << "[MergePatternStructures] Added connecting edge: " 
                 << connecting_from_label << " -> " << connecting_to_label << "\n";
  }
  
  return merged_pattern;
}

// Helper function to get operation label (matches DFGExtractor logic)
std::string GraMi::getOperationLabel(mlir::Operation* op) {
  std::string opName = op->getName().getStringRef().str();
  
  // Remove dialect prefix
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

// Merge two pattern instances into one
PatternInstance GraMi::mergePatternInstances(const PatternInstance& instance1, 
                                            const PatternInstance& instance2) {
  
  PatternInstance merged_instance;
  
  // Combine operations from both instances and sort them in topological order
  std::vector<mlir::Operation*> all_ops;
  all_ops.insert(all_ops.end(), instance1.operations.begin(), instance1.operations.end());
  all_ops.insert(all_ops.end(), instance2.operations.begin(), instance2.operations.end());
  
  // Sort operations in topological order based on their positions in the block
  std::sort(all_ops.begin(), all_ops.end(), [](mlir::Operation* a, mlir::Operation* b) {
    return a->isBeforeInBlock(b);
  });
  
  merged_instance.operations = all_ops;
  
  // Set the last operation (the one that comes later in the block)
  if (instance1.lastOp && instance2.lastOp) {
    if (instance1.lastOp->isBeforeInBlock(instance2.lastOp)) {
      merged_instance.lastOp = instance2.lastOp;
    } else {
      merged_instance.lastOp = instance1.lastOp;
    }
  } else if (instance1.lastOp) {
    merged_instance.lastOp = instance1.lastOp;
  } else {
    merged_instance.lastOp = instance2.lastOp;
  }
  
  // Combine inputs (external inputs to the merged pattern)
  llvm::DenseSet<mlir::Operation*> merged_ops(
      merged_instance.operations.begin(), merged_instance.operations.end());
  
  llvm::SetVector<mlir::Value> input_set;
  
  for (mlir::Operation* op : merged_instance.operations) {
    for (mlir::Value operand : op->getOperands()) {
      mlir::Operation* def_op = operand.getDefiningOp();
      if (!def_op || !merged_ops.contains(def_op)) {
        input_set.insert(operand);
      }
    }
  }
  
  merged_instance.inputs = std::vector<mlir::Value>(input_set.begin(), input_set.end());
  
  // Combine outputs (results used outside the merged pattern)
  llvm::SetVector<mlir::Value> output_set;
  
  for (mlir::Operation* op : merged_instance.operations) {
    for (mlir::Value result : op->getResults()) {
      bool has_external_use = false;
      for (mlir::OpOperand& use : result.getUses()) {
        mlir::Operation* user = use.getOwner();
        if (!merged_ops.contains(user)) {
          has_external_use = true;
          break;
        }
      }
      
      if (has_external_use) {
        output_set.insert(result);
      }
    }
  }
  
  merged_instance.outputs = std::vector<mlir::Value>(output_set.begin(), output_set.end());
  
  // Set pattern properties
  merged_instance.pattern_id = -1; // Will be set later
  merged_instance.frequency = 1;
  
  return merged_instance;
}

// Generate a readable string representation of the merged pattern
std::string GraMi::generateMergedPatternString(const FrequentSubgraph& merged_pattern) {
  std::ostringstream oss;
  oss << "MergedPattern[";
  oss << "Nodes: " << merged_pattern.getNodes().size();
  oss << ", Edges: " << merged_pattern.getEdges().size();
  oss << ", Frequency: " << merged_pattern.getFrequency();
  oss << "]";
  return oss.str();
}
