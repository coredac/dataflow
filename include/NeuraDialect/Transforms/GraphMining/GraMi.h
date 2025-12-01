#ifndef NEURA_GRAMI_H
#define NEURA_GRAMI_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <cstdint>

namespace mlir::neura {

// Forward declarations
class DfgNode;
class DfgEdge;
class DfgGraph;
class FrequentSubgraph;

// DFG Node representing an operation
class DfgNode {
public:
  using NodeId = size_t;
  
  DfgNode(NodeId id, mlir::Operation* op, const std::string& label)
    : id_(id), operation_(op), label_(label) {}
  
  NodeId getId() const { return id_; }
  mlir::Operation* getOperation() const { return operation_; }
  const std::string& getLabel() const { return label_; }
  
  // Add edge to this node
  void addIncomingEdge(DfgEdge* edge) { incoming_edges_.push_back(edge); }
  void addOutgoingEdge(DfgEdge* edge) { outgoing_edges_.push_back(edge); }
  
  const std::vector<DfgEdge*>& getIncomingEdges() const { return incoming_edges_; }
  const std::vector<DfgEdge*>& getOutgoingEdges() const { return outgoing_edges_; }
  
private:
  NodeId id_;
  mlir::Operation* operation_;
  std::string label_;
  std::vector<DfgEdge*> incoming_edges_;
  std::vector<DfgEdge*> outgoing_edges_;
};

// DFG Edge representing data flow between operations
class DfgEdge {
public:
  using EdgeId = size_t;
  
  DfgEdge(EdgeId id, DfgNode* from, DfgNode* to, mlir::Value value)
    : id_(id), from_(from), to_(to), value_(value) {}
  
  EdgeId getId() const { return id_; }
  DfgNode* getFrom() const { return from_; }
  DfgNode* getTo() const { return to_; }
  mlir::Value getValue() const { return value_; }
  
private:
  EdgeId id_;
  DfgNode* from_;
  DfgNode* to_;
  mlir::Value value_;
};

// DFG Graph representing the entire data flow graph
class DfgGraph {
public:
  DfgGraph() : next_node_id_(0), next_edge_id_(0) {}
  
  // Add node to the graph
  DfgNode* addNode(mlir::Operation* op, const std::string& label);
  
  // Add edge to the graph
  DfgEdge* addEdge(DfgNode* from, DfgNode* to, mlir::Value value);
  
  // Get all nodes
  const std::vector<DfgNode*>& getNodes() const { return nodes_; }
  
  // Get all edges
  const std::vector<DfgEdge*>& getEdges() const { return edges_; }
  
  // Get node by ID
  DfgNode* getNode(DfgNode::NodeId id) const;
  
  // Get edge by ID
  DfgEdge* getEdge(DfgEdge::EdgeId id) const;
  
  // Get number of nodes and edges
  size_t getNumNodes() const { return nodes_.size(); }
  size_t getNumEdges() const { return edges_.size(); }
  
  // Clear the graph
  void clear();
  
private:
  std::vector<DfgNode*> nodes_;
  std::vector<DfgEdge*> edges_;
  llvm::DenseMap<mlir::Operation*, DfgNode*> op_to_node_;
  DfgNode::NodeId next_node_id_;
  DfgEdge::EdgeId next_edge_id_;
};

// Frequent subgraph pattern
class FrequentSubgraph {
public:
  using NodeId = DfgNode::NodeId;
  using EdgeId = DfgEdge::EdgeId;
  
  FrequentSubgraph(const std::string& pattern, size_t frequency, int64_t id = -1)
    : pattern_(pattern), frequency_(frequency), id_(id) {}
  
  // Copy constructor with new frequency (using tag to disambiguate from copy constructor)
  FrequentSubgraph(const FrequentSubgraph& other, size_t new_frequency)
    : pattern_(other.pattern_), frequency_(new_frequency), id_(other.id_),
      nodes_(other.nodes_), edges_(other.edges_) {}
  
  const std::string& getPattern() const { return pattern_; }
  size_t getFrequency() const { return frequency_; }
  int64_t getId() const { return id_; }
  void setId(int64_t id) { id_ = id; }
  
  // Add node to the pattern
  void addNode(NodeId node_id, const std::string& label) {
    nodes_[node_id] = label;
  }
  
  // Add edge to the pattern
  void addEdge(EdgeId edge_id, NodeId from, NodeId to) {
    edges_[edge_id] = std::make_pair(from, to);
  }
  
  const std::map<NodeId, std::string>& getNodes() const { return nodes_; }
  const std::map<EdgeId, std::pair<NodeId, NodeId>>& getEdges() const { return edges_; }
  
private:
  std::string pattern_;
  size_t frequency_;
  int64_t id_;
  std::map<NodeId, std::string> nodes_;
  std::map<EdgeId, std::pair<NodeId, NodeId>> edges_;
};

// Pattern instance in the actual code
struct PatternInstance {
  std::vector<mlir::Operation*> operations;
  std::vector<mlir::Value> inputs;   // External inputs to the pattern  
  std::vector<mlir::Value> outputs;  // Outputs from the pattern
  mlir::Operation* last_op = nullptr; // Last operation in the pattern
  int64_t pattern_id;
  size_t frequency;  // Weight for MWIS
};

// Pattern with all its instances
struct PatternWithInstances {
  int64_t pattern_id;
  size_t frequency;  // Weight for MWIS
  std::vector<PatternInstance> instances;
};

// Result from GraMi mining: pattern + selected instances
struct PatternWithSelectedInstances {
  FrequentSubgraph pattern;
  std::vector<PatternInstance> selected_instances;
  PatternWithSelectedInstances(const FrequentSubgraph& p) : pattern(p) {}
};

// GraMi algorithm implementation
class GraMi {
public:
  GraMi(DfgGraph* graph, size_t min_support = 2)
    : graph_(graph), min_support_(min_support) {}
  
  // Main mining function
  std::vector<PatternWithSelectedInstances> mineFrequentSubgraphs();
  
  // Maximum Weighted Independent Set selection for instances
  // Input: all instances (with pattern_id set), frequent subgraphs, min support
  // Output: patterns with selected instances that don't conflict
  static std::vector<PatternWithSelectedInstances> selectMaxWeightedIndependentSetForInstances(
      const std::vector<PatternInstance>& instances,
      const std::vector<FrequentSubgraph>& frequent_subgraphs,
      size_t min_support);
  
  // Set minimum support threshold
  void setMinSupport(size_t min_support) { min_support_ = min_support; }
  
private:
  DfgGraph* graph_;
  size_t min_support_;
  
  // Helper functions for GraMi algorithm
  std::vector<FrequentSubgraph> generateCandidates();
  bool isFrequent(const FrequentSubgraph& candidate);
  std::string generatePatternString(const FrequentSubgraph& subgraph);
  
  // Support counting
  size_t countSupport(const FrequentSubgraph& pattern);
  
  // Helper functions for MWIS
  static bool instancesConflict(const PatternInstance& a, const PatternInstance& b);
  static bool patternsConflict(const PatternWithInstances& a, const PatternWithInstances& b);
  
private:
  // Helper function to get operation label (matches DfgExtractor logic)
  std::string getOperationLabel(mlir::Operation* op);
};

// Utility functions for graph extraction from MLIR
class DfgExtractor {
public:
  // Extract DFG from MLIR module
  static std::unique_ptr<DfgGraph> extractFromModule(mlir::ModuleOp module);
  
  // Extract DFG from MLIR function
  static std::unique_ptr<DfgGraph> extractFromFunction(mlir::func::FuncOp func);
  
  // Extract DFG from MLIR block
  static std::unique_ptr<DfgGraph> extractFromBlock(mlir::Block* block);
  
private:
  // Helper functions
  static std::string getOperationLabel(mlir::Operation* op);
  static bool shouldIncludeOperation(mlir::Operation* op);
};

} // namespace mlir::neura

#endif // NEURA_GRAMI_H
