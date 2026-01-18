#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/Transforms/GraphMining/GraMi.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace mlir::neura;

// Static member definition for tracking attempted patterns
std::set<std::string> GraMi::attempted_patterns_;

DfgNode* DfgGraph::addNode(mlir::Operation* op, const std::string& label) {
  auto node = new DfgNode(next_node_id_++, op, label);
  nodes_.push_back(node);
  op_to_node_[op] = node;
  return node;
}

DfgEdge* DfgGraph::addEdge(DfgNode* from, DfgNode* to, mlir::Value value) {
  auto edge = new DfgEdge(next_edge_id_++, from, to, value);
  edges_.push_back(edge);
  from->addOutgoingEdge(edge);
  to->addIncomingEdge(edge);
  return edge;
}

DfgNode* DfgGraph::getNode(DfgNode::NodeId id) const {
  if (id < nodes_.size()) {
    return nodes_[id];
  }
  return nullptr;
}

DfgEdge* DfgGraph::getEdge(DfgEdge::EdgeId id) const {
  if (id < edges_.size()) {
    return edges_[id];
  }
  return nullptr;
}

void DfgGraph::clear() {
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

std::string DfgExtractor::getOperationLabel(mlir::Operation* op) {
  std::string op_name = op->getName().getStringRef().str();
  
  size_t dot_pos = op_name.find('.');
  if (dot_pos != std::string::npos) {
    op_name = op_name.substr(dot_pos + 1);
  }
  
  if (op->getNumResults() > 0) {
    Type result_type = op->getResult(0).getType();
    if (auto int_type = mlir::dyn_cast<IntegerType>(result_type)) {
      op_name += "_i" + std::to_string(int_type.getWidth());
    } else if (auto float_type = mlir::dyn_cast<FloatType>(result_type)) {
      op_name += "_f" + std::to_string(float_type.getWidth());
    }
  }
  
  return op_name;
}

// Excludes operations that are not part of the DFG since they don't involve computation and will not be mapped onto the functional units.
bool DfgExtractor::shouldIncludeOperation(mlir::Operation* op) {
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
  
  if (op->getDialect()->getNamespace() == "neura") {
    return true;
  }
  
  if (op->getDialect()->getNamespace() == "llvm") {
    return false;
  }
  
  if (op->getDialect()->getNamespace() == "arith") {
    return false;
  }
  
  llvm::errs() << "Excluding operation: " << op->getName().getStringRef() << "\n";

  return false;
}

// Extracts the data flow graph from the module.
std::unique_ptr<DfgGraph> DfgExtractor::extractFromModule(ModuleOp module) {
  auto graph = std::make_unique<DfgGraph>();
  
  module.walk([&](func::FuncOp func) {
    llvm::errs() << "Extracting DFG from function: " << func.getName() << "\n";

    auto func_graph = extractFromFunction(func);
    if (func_graph) {
      for (auto* node : func_graph->getNodes()) {
        graph->addNode(node->getOperation(), node->getLabel());
      }
      for (auto* edge : func_graph->getEdges()) {
        graph->addEdge(edge->getFrom(), edge->getTo(), edge->getValue());
      }
    }
  });
  
  return graph;
}

// Extracts the data flow graph from the function.
std::unique_ptr<DfgGraph> DfgExtractor::extractFromFunction(func::FuncOp func) {
  auto graph = std::make_unique<DfgGraph>();
  
  func.walk([&](Block* block) {
    auto block_graph = extractFromBlock(block);
    if (block_graph) {
      for (auto* node : block_graph->getNodes()) {
        graph->addNode(node->getOperation(), node->getLabel());
      }
      for (auto* edge : block_graph->getEdges()) {
        graph->addEdge(edge->getFrom(), edge->getTo(), edge->getValue());
      }
    }
  });
  
  return graph;
}

// Extracts the data flow graph from the block.
std::unique_ptr<DfgGraph> DfgExtractor::extractFromBlock(mlir::Block* block) {
  auto graph = std::make_unique<DfgGraph>();
  llvm::DenseMap<mlir::Value, DfgNode*> value_to_node;
  
  for (auto& op : block->getOperations()) {
    if (shouldIncludeOperation(&op)) {
      std::string label = getOperationLabel(&op);
      DfgNode* node = graph->addNode(&op, label);
      
      for (mlir::Value result : op.getResults()) {
        value_to_node[result] = node;
      }
    }
  }
  
  for (auto& op : block->getOperations()) {
    if (shouldIncludeOperation(&op)) {
      DfgNode* current_node = nullptr;
      
      for (mlir::Value result : op.getResults()) {
        if (value_to_node.count(result)) {
          current_node = value_to_node[result];
          break;
        }
      }
      
      if (!current_node) continue;
      
      for (mlir::Value operand : op.getOperands()) {
        if (value_to_node.count(operand)) {
          DfgNode* source_node = value_to_node[operand];
          graph->addEdge(source_node, current_node, operand);
        }
      }
    }
  }
  
  return graph;
}

// Mines the frequent subgraphs from the data flow graph.
// Algorithm:
// 1. Collects all 2-node patterns from the graph
// 2. For each pattern, separates instances into critical path vs non-critical path
// 3. For each pattern, performs MWIS with higher weight for critical path instances
// 4. Performs inter-pattern analysis with critical path conflict priority
std::vector<PatternWithSelectedInstances> GraMi::mineFrequentSubgraphs() {
  std::vector<FrequentSubgraph> frequent_subgraphs;
  
  // Map from pattern string to (critical instances, non-critical instances)
  std::map<std::string, std::pair<std::vector<PatternInstance>, std::vector<PatternInstance>>> pattern_instances;
  
  auto derive_label = [](mlir::Operation* op, const std::string& fallback_label) -> std::string {
    if (!op) return fallback_label;
    auto name = op->getName().getStringRef();
    if (name.ends_with(attr::val::kOpFused) || name.contains(attr::val::kNeuraFusedOp)) {
      if (auto attr = op->getAttr("pattern_name")) {
        if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
          return std::string("fused_op:") + str_attr.getValue().str();
        }
      }
      return std::string(attr::val::kOpFused);
    }
    return fallback_label;
  };

  llvm::errs() << "[GraMi] Critical path ops count: " << critical_path_ops_.size() << "\n";

  // Step 1: Collects all 2-node patterns and classifies instances
  for (auto* edge : graph_->getEdges()) {
    DfgNode* from = edge->getFrom();
    DfgNode* to = edge->getTo();

    auto* from_op = from->getOperation();
    auto* to_op = to->getOperation();


    // Skips operations inside fused_op
    if (from_op->getParentRegion()->getParentOp()->getName().getStringRef().str() == "neura.fused_op" || to_op->getParentRegion()->getParentOp()->getName().getStringRef().str() == "neura.fused_op") {
      continue;
    }

    std::string from_label = derive_label(from_op, from->getLabel());
    std::string to_label = derive_label(to_op, to->getLabel());
    std::string pattern = from_label + "->" + to_label;
    
    PatternInstance instance;
    instance.frequency = 1;
    
    if (from_op->isBeforeInBlock(to_op)) {
      instance.operations.push_back(from_op);
      instance.operations.push_back(to_op);
      instance.last_op = to_op;
    } else {
      instance.operations.push_back(to_op);
      instance.operations.push_back(from_op);
      instance.last_op = from_op;
    }
    
    llvm::DenseSet<mlir::Operation*> pattern_ops;
    pattern_ops.insert(from_op);
    pattern_ops.insert(to_op);
    
    llvm::SetVector<mlir::Value> input_set;
    for (mlir::Value operand : from_op->getOperands()) {
      input_set.insert(operand);
    }
    for (mlir::Value operand : to_op->getOperands()) {
      if (operand.getDefiningOp() != from_op) {
        input_set.insert(operand);
      }
    }
    instance.inputs = std::vector<mlir::Value>(input_set.begin(), input_set.end());
    
    llvm::SetVector<mlir::Value> output_set;
    for (mlir::Operation* op : instance.operations) {
      for (mlir::Value result : op->getResults()) {
        bool has_external_use = false;
        for (mlir::OpOperand& use : result.getUses()) {
          mlir::Operation* user = use.getOwner();
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
    instance.outputs = std::vector<mlir::Value>(output_set.begin(), output_set.end());
    
    instance.is_on_critical_path = isInstanceOnCriticalPath(instance);
    
    if (instance.is_on_critical_path) {
      pattern_instances[pattern].first.push_back(instance);
    } else {
      pattern_instances[pattern].second.push_back(instance);
    }
  }

  // Step 2: Processes frequent patterns and performs per-pattern MWIS
  std::vector<PatternWithSelectedInstances> candidates;
  
  for (auto& [pattern, instances_pair] : pattern_instances) {
    auto& [critical_instances, non_critical_instances] = instances_pair;
    size_t total_count = critical_instances.size() + non_critical_instances.size();
    
    // Skips patterns that have been attempted for fusion
    if (hasPatternBeenAttempted(pattern)) {
      continue;
    }
    
    if (total_count >= min_support_) {
      size_t pattern_idx = frequent_subgraphs.size();
      std::string from_label = pattern.substr(0, pattern.find("->"));
      std::string to_label = pattern.substr(pattern.find("->") + 2);
      FrequentSubgraph subgraph(pattern, total_count, static_cast<int64_t>(pattern_idx));
      subgraph.addNode(0, from_label);
      subgraph.addNode(1, to_label);
      subgraph.addEdge(0, 0, 1);
      frequent_subgraphs.push_back(subgraph);
      
      for (auto& inst : critical_instances) {
        inst.pattern_id = static_cast<int64_t>(pattern_idx);
      }
      for (auto& inst : non_critical_instances) {
        inst.pattern_id = static_cast<int64_t>(pattern_idx);
      }
      
      auto [selected_critical, selected_non_critical] = selectMWISForPattern(critical_instances, non_critical_instances, 10.0);
      
      // Creates PatternWithSelectedInstances
      PatternWithSelectedInstances pwsi(subgraph);
      pwsi.critical_instances = selected_critical;
      pwsi.non_critical_instances = selected_non_critical;
      pwsi.selected_instances.insert(pwsi.selected_instances.end(), selected_critical.begin(), selected_critical.end());
      pwsi.selected_instances.insert(pwsi.selected_instances.end(), selected_non_critical.begin(), selected_non_critical.end());
      
      candidates.push_back(pwsi);
      
      llvm::errs() << "[GraMi] Pattern #" << pattern_idx << " after intra-pattern MWIS: " << selected_critical.size() << " critical, " << selected_non_critical.size() << " non-critical selected\n";
    }
  }

  if (candidates.empty()) {
    llvm::errs() << "[GraMi] No frequent patterns found\n";
    return {};
  }
  
  // Step 3: Performs inter-pattern analysis with critical path priority
  std::vector<PatternWithSelectedInstances> result = selectPatternsWithCriticalPriority(candidates, min_support_);
  
  llvm::errs() << "[GraMi] Final result: " << result.size() << " patterns\n";

  // Prints summary
  size_t total_critical = 0, total_non_critical = 0;
  for (const auto& p : result) {
    total_critical += p.critical_instances.size();
    total_non_critical += p.non_critical_instances.size();
  }
  llvm::errs() << "[GraMi] Summary: " << result.size() << " patterns, " << total_critical << " critical instances, " << total_non_critical << " non-critical instances\n";

  return result;
}

// Checks if the candidate pattern is frequent using the threshold min_support_.
bool GraMi::isFrequent(const FrequentSubgraph& candidate) {
  size_t support = countSupport(candidate);
  return support >= min_support_;
}

// Counts the support of the pattern in the data flow graph.
size_t GraMi::countSupport(const FrequentSubgraph& pattern) {
  std::map<std::string, size_t> node_counts;
  for (const auto& pair : pattern.getNodes()) {
    node_counts[pair.second]++;
  }
  
  std::map<std::string, size_t> graph_node_counts;
  for (auto* node : graph_->getNodes()) {
    graph_node_counts[node->getLabel()]++;
  }
  
  size_t min_count = SIZE_MAX;
  for (const auto& pair : node_counts) {
    size_t graph_count = graph_node_counts[pair.first];
    size_t required_count = pair.second;
    if (graph_count < required_count) {
      return 0;
    }
    min_count = std::min(min_count, graph_count / required_count);
  }
  
  return min_count;
}

// Generates a string representation of the pattern.
std::string GraMi::generatePatternString(const FrequentSubgraph& subgraph) {
  std::ostringstream oss;
  oss << "Pattern: ";
  
  oss << "Nodes[";
  for (const auto& pair : subgraph.getNodes()) {
    oss << pair.first << ":" << pair.second << " ";
  }
  oss << "] ";
  
  oss << "Edges[";
  for (const auto& pair : subgraph.getEdges()) {
    oss << pair.first << ":" << pair.second.first << "->" << pair.second.second << " ";
  }
  oss << "] ";
  
  oss << "Support: " << subgraph.getFrequency();
  
  return oss.str();
}

// Collects critical path operations from the function.
// Critical paths are recurrence cycles with maximum length.
llvm::DenseSet<mlir::Operation*> GraMi::collectCriticalPathOps(mlir::func::FuncOp func) {
  llvm::DenseSet<mlir::Operation*> critical_ops;
  
  // Collects all recurrence cycles
  auto recurrence_cycles = collectRecurrenceCycles(func);
  
  if (recurrence_cycles.empty()) {
    llvm::errs() << "[GraMi] No recurrence cycles found\n";
    return critical_ops;
  }
  
  // Finds the maximum recurrence length
  int max_length = 0;
  for (const auto& cycle : recurrence_cycles) {
    max_length = std::max(max_length, cycle.length);
  }
  
  llvm::errs() << "[GraMi] Maximum recurrence length: " << max_length << "\n";
  
  // Collects operations from all cycles with maximum length
  int critical_cycle_count = 0;
  for (const auto& cycle : recurrence_cycles) {
    if (cycle.length == max_length) {
      critical_cycle_count++;
      for (mlir::Operation* op : cycle.operations) {
        critical_ops.insert(op);
      }
      llvm::errs() << "[GraMi] Critical path cycle (length " << cycle.length << "):\n";
      for (mlir::Operation* op : cycle.operations) {
        llvm::errs() << "  " << *op << "\n";
      }
    }
  }
  
  llvm::errs() << "[GraMi] Found " << critical_cycle_count << " critical path(s) with " 
               << critical_ops.size() << " total operations\n";
  
  return critical_ops;
}

// Checks if an instance is on critical path (all operations of the instance must be on critical path)
bool GraMi::isInstanceOnCriticalPath(const PatternInstance& instance) const {
  for (mlir::Operation* op : instance.operations) {
    if (!critical_path_ops_.contains(op)) {
      return false;
    }
  }
  return true;
}

// Checks if the two instances conflict. Conflict occurs if the two instances have the same operation.
bool GraMi::instancesConflict(const PatternInstance& a, const PatternInstance& b) {
  for (mlir::Operation* op_a : a.operations) {
    for (mlir::Operation* op_b : b.operations) {
      if (op_a == op_b) return true;
    }
  }
  return false;
}

// Checks if the two patterns conflict. If any instance in the two patterns conflict, the patterns conflict.
bool GraMi::patternsConflict(const PatternWithInstances& a, const PatternWithInstances& b) {
  for (const auto& inst_a : a.instances) {
    for (const auto& inst_b : b.instances) {
      if (instancesConflict(inst_a, inst_b)) {
        return true;
      }
    }
  }
  return false;
}

// Checks if two patterns have conflicting critical path instances
bool GraMi::criticalInstancesConflict(const PatternWithSelectedInstances& a, 
                                      const PatternWithSelectedInstances& b) {
  for (const auto& inst_a : a.critical_instances) {
    for (const auto& inst_b : b.critical_instances) {
      if (instancesConflict(inst_a, inst_b)) {
        return true;
      }
    }
  }
  return false;
}

// Selects maximum weighted independent set for a single pattern
// Critical path instances have higher weight
std::pair<std::vector<PatternInstance>, std::vector<PatternInstance>> GraMi::selectMWISForPattern(const std::vector<PatternInstance>& critical_instances, const std::vector<PatternInstance>& non_critical_instances, double critical_weight_multiplier) {
  
  // Combines all instances with their weights
  std::vector<std::pair<PatternInstance, double>> weighted_instances;
  
  for (const auto& inst : critical_instances) {
    weighted_instances.push_back({inst, critical_weight_multiplier});
  }
  for (const auto& inst : non_critical_instances) {
    weighted_instances.push_back({inst, 1.0});
  }
  
  if (weighted_instances.empty()) {
    return {{}, {}};
  }
  
  size_t n = weighted_instances.size();
  
  // Builds conflict graph
  std::vector<std::vector<size_t>> conflicts(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (instancesConflict(weighted_instances[i].first, weighted_instances[j].first)) {
        conflicts[i].push_back(j);
        conflicts[j].push_back(i);
      }
    }
  }
  
  // Greedy MWIS selection: prioritizes by weight / (degree + 1)
  std::vector<size_t> selected_indices;
  std::vector<bool> available(n, true);
  
  while (true) {
    size_t best_idx = n;
    double best_score = -1.0;
    
    for (size_t i = 0; i < n; ++i) {
      if (!available[i]) continue;
      
      size_t active_degree = 0;
      for (size_t neighbor : conflicts[i]) {
        if (available[neighbor]) active_degree++;
      }
      
      double score = weighted_instances[i].second / (active_degree + 1);
      
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
    
    if (best_idx == n) break;
    
    selected_indices.push_back(best_idx);
    available[best_idx] = false;
    
    for (size_t neighbor : conflicts[best_idx]) {
      available[neighbor] = false;
    }
  }
  
  // Separates selected instances into critical and non-critical
  std::vector<PatternInstance> selected_critical;
  std::vector<PatternInstance> selected_non_critical;
  
  size_t critical_count = critical_instances.size();
  for (size_t idx : selected_indices) {
    if (idx < critical_count) {
      selected_critical.push_back(weighted_instances[idx].first);
    } else {
      selected_non_critical.push_back(weighted_instances[idx].first);
    }
  }
  
  return {selected_critical, selected_non_critical};
}

// Inter-pattern independent set analysis with critical path priority
// Rules:
// - If two patterns have conflicting critical instances, they cannot coexist
//   Chooses the pattern with more critical instances
// - Non-critical vs non-critical or non-critical vs critical conflicts are allowed
std::vector<PatternWithSelectedInstances> GraMi::selectPatternsWithCriticalPriority(std::vector<PatternWithSelectedInstances>& candidates, size_t min_support) {
  
  if (candidates.empty()) return {};
  
  // Sorts candidates by number of critical instances (descending), then by total instances
  std::sort(candidates.begin(), candidates.end(),
    [](const PatternWithSelectedInstances& a, const PatternWithSelectedInstances& b) {
      if (a.critical_instances.size() != b.critical_instances.size()) {
        return a.critical_instances.size() > b.critical_instances.size();
      }
      return a.selected_instances.size() > b.selected_instances.size();
    });
  
  std::vector<PatternWithSelectedInstances> result;
  
  for (size_t i = 0; i < candidates.size(); ++i) {
    // Checks for critical instance conflicts with already selected patterns
    bool has_critical_conflict = false;
    for (const auto& selected : result) {
      if (criticalInstancesConflict(candidates[i], selected)) {
        has_critical_conflict = true;
        break;
      }
    }
    
    if (has_critical_conflict) {
      continue;
    }
    
    result.push_back(candidates[i]);
  }
  
  // Now handles non-critical conflicts: removes conflicting non-critical instances
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      // Finds non-critical instances in pattern j that conflict with pattern i
      std::vector<PatternInstance> remaining_non_critical;
      for (const auto& inst_j : result[j].non_critical_instances) {
        bool conflicts_with_i = false;
        // Checks conflict with pattern i
        for (const auto& inst_i : result[i].selected_instances) {
          if (instancesConflict(inst_i, inst_j)) {
            conflicts_with_i = true;
            break;
          }
        }
        if (!conflicts_with_i) {
          remaining_non_critical.push_back(inst_j);
        }
      }
      result[j].non_critical_instances = remaining_non_critical;
      
      // Updates selected_instances
      result[j].selected_instances.clear();
      result[j].selected_instances.insert(result[j].selected_instances.end(), result[j].critical_instances.begin(), result[j].critical_instances.end());
      result[j].selected_instances.insert(result[j].selected_instances.end(), result[j].non_critical_instances.begin(), result[j].non_critical_instances.end());
    }
  }
  
  return result;
}

// Gets the label of the operation.
std::string GraMi::getOperationLabel(mlir::Operation* op) {
  std::string op_name = op->getName().getStringRef().str();
  
  size_t dot_pos = op_name.find('.');
  if (dot_pos != std::string::npos) {
    op_name = op_name.substr(dot_pos + 1);
  }
  
  if (op->getNumResults() > 0) {
    Type result_type = op->getResult(0).getType();
    if (auto int_type = mlir::dyn_cast<IntegerType>(result_type)) {
      op_name += "_i" + std::to_string(int_type.getWidth());
    } else if (auto float_type = mlir::dyn_cast<FloatType>(result_type)) {
      op_name += "_f" + std::to_string(float_type.getWidth());
    }
  }
  
  return op_name;
}

// Checks if a pattern has been attempted for fusion
bool GraMi::hasPatternBeenAttempted(const std::string& pattern) {
  return attempted_patterns_.find(pattern) != attempted_patterns_.end();
}

// Marks a pattern as attempted for fusion
void GraMi::markPatternAsAttempted(const std::string& pattern) {
  attempted_patterns_.insert(pattern);
  llvm::errs() << "[GraMi] Marked pattern as attempted: " << pattern << "\n";
}

// Clears all attempted pattern marks
void GraMi::clearAttemptedPatterns() {
  attempted_patterns_.clear();
  llvm::errs() << "[GraMi] Cleared all attempted pattern marks\n";
}
