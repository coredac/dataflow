#include "NeuraDialect/Transforms/GraphMining/GraMi.h"
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
std::vector<PatternWithSelectedInstances> GraMi::mineFrequentSubgraphs() {
  std::vector<FrequentSubgraph> frequent_subgraphs;
  
  std::map<std::string, std::vector<PatternInstance>> pattern_instances;
  
  auto derive_label = [](mlir::Operation* op, const std::string& fallback_label) -> std::string {
    if (!op) return fallback_label;
    auto name = op->getName().getStringRef();
    if (name.ends_with("fused_op") || name.contains("neura.fused_op")) {
      if (auto attr = op->getAttr("pattern_name")) {
        if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(attr)) {
          return std::string("fused_op:") + str_attr.getValue().str();
        }
      }
      return std::string("fused_op");
    }
    return fallback_label;
  };

  for (auto* edge : graph_->getEdges()) {
    DfgNode* from = edge->getFrom();
    DfgNode* to = edge->getTo();

    auto* from_op = from->getOperation();
    auto* to_op = to->getOperation();

    if (from_op->getParentRegion() == to_op->getParentRegion() && from_op->getParentRegion()->getParentOp()->getName().getStringRef().str() == "neura.fused_op") {
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
    
    pattern_instances[pattern].push_back(instance);
  }

  std::vector<PatternInstance> all_instances;
  
  for (auto& [pattern, instances] : pattern_instances) {
    if (instances.size() >= min_support_) {
      size_t pattern_idx = frequent_subgraphs.size();
      std::string from_label = pattern.substr(0, pattern.find("->"));
      std::string to_label = pattern.substr(pattern.find("->") + 2);
      FrequentSubgraph subgraph(pattern, instances.size(), static_cast<int64_t>(pattern_idx));
      subgraph.addNode(0, from_label);
      subgraph.addNode(1, to_label);
      subgraph.addEdge(0, 0, 1);
      frequent_subgraphs.push_back(subgraph);
      
      for (auto& instance : instances) {
        instance.pattern_id = static_cast<int64_t>(pattern_idx);
        all_instances.push_back(instance);
      }
      
      llvm::errs() << "[GraMi] Pattern #" << pattern_idx << " (" << pattern << "): " << instances.size() << " total instances\n";
    }
  }

  if (frequent_subgraphs.empty()) {
    return {};
  }
  
  llvm::errs() << "[GraMi] Total instances across all patterns: " << all_instances.size() << "\n";
  
  std::vector<PatternWithSelectedInstances> result = selectMaxWeightedIndependentSetForInstances(all_instances, frequent_subgraphs, min_support_);
  
  llvm::errs() << "[GraMi] Final result: " << result.size() << " patterns\n";

  llvm::errs() << "[GraMi] Verifying no conflicts in selected instances...\n";
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      for (const auto& inst_i : result[i].selected_instances) {
        for (const auto& inst_j : result[j].selected_instances) {
          if (instancesConflict(inst_i, inst_j)) {
            llvm::errs() << "[GraMi ERROR] Conflict detected between selected instances!\n";
            llvm::errs() << "  Pattern " << i << " and Pattern " << j << "\n";
            llvm::errs() << "  Conflicting operations found\n";
            assert(false && "GraMi global MWIS selection produced conflicting instances.");
          }
        }
      }
    }
  }
  llvm::errs() << "[GraMi] Verification passed: No conflicts detected.\n";

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

// Selects the maximum weighted independent set of instances from the frequent subgraphs. The independent set is a set of instances that do not conflict with each other.
std::vector<PatternWithSelectedInstances> GraMi::selectMaxWeightedIndependentSetForInstances(const std::vector<PatternInstance>& instances, const std::vector<FrequentSubgraph>& frequent_subgraphs, size_t min_support) {
  if (instances.empty()) return {};
  
  size_t n = instances.size();
  
  std::vector<std::vector<size_t>> conflicts(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (instancesConflict(instances[i], instances[j])) {
        conflicts[i].push_back(j);
        conflicts[j].push_back(i);
      }
    }
  }
  
  std::vector<size_t> selected_indices;
  std::vector<bool> available(n, true);
  
  while (true) {
    size_t best_idx = n;
    double best_ratio = -1.0;
    
    for (size_t i = 0; i < n; ++i) {
      if (!available[i]) continue;
      
      size_t active_degree = 0;
      for (size_t neighbor : conflicts[i]) {
        if (available[neighbor]) active_degree++;
      }
      
      double ratio = 1.0 / (active_degree + 1);
      
      if (ratio > best_ratio) {
        best_ratio = ratio;
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
  
  llvm::errs() << "[GraMi] Selected " << selected_indices.size() << " non-conflicting instances globally\n";
  
  std::map<size_t, std::vector<PatternInstance>> pattern_to_instances;
  for (size_t idx : selected_indices) {
    size_t pattern_idx = static_cast<size_t>(instances[idx].pattern_id);
    pattern_to_instances[pattern_idx].push_back(instances[idx]);
  }
  
  std::vector<PatternWithSelectedInstances> result;
  for (const auto& [pattern_idx, selected_instances] : pattern_to_instances) {
    if (selected_instances.size() >= min_support) {
      const auto& orig_subgraph = frequent_subgraphs[pattern_idx];
      
      FrequentSubgraph updated_subgraph(orig_subgraph, selected_instances.size());
      
      PatternWithSelectedInstances pwsi(updated_subgraph);
      pwsi.selected_instances = selected_instances;
      
      result.push_back(pwsi);
      
      llvm::errs() << "[GraMi] Pattern #" << pattern_idx << " (" << orig_subgraph.getPattern() << "): " << selected_instances.size() << " non-conflicting instances\n";
    }
  }
  
  llvm::errs() << "[GraMi] Found " << result.size() << " patterns with sufficient frequency (min_support=" << min_support << ")\n";
  
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
