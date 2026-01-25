//===- HardwareTemplate.h - Hardware Template Data Structures and Helpers -===//
//
// This file contains declarations for hardware template data structures and
// helper functions for hardware template merging.
//
// The hardware template system maximizes pattern coverage while minimizing
// hardware cost through resource sharing. Key concepts:
//
// - Functional Unit (FU): A single hardware unit that executes one operation type
// - HardwareTemplate: A collection of FUs with connections supporting multiple patterns
// - HardwarePattern: A sequence of operations mapped to template FUs
//
// For detailed documentation with examples and diagrams, see:
//   docs/HardwareTemplateGuide.md
//
//===----------------------------------------------------------------------===//

#ifndef NEURA_DIALECT_TRANSFORMS_GRAPHMINING_HARDWARETEMPLATE_H
#define NEURA_DIALECT_TRANSFORMS_GRAPHMINING_HARDWARETEMPLATE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cstdint>
#include <utility>

namespace mlir {
namespace neura {
class FusedOp;
}
}

namespace mlir::neura {

// Forward declarations
struct HardwarePattern {
  int64_t id;
  std::string name;
  int64_t freq;
  std::vector<std::string> ops;
  std::vector<int> op_levels;  // Topological level for each op (ops at same level can run in parallel)
  std::vector<std::vector<int>> op_preds;  // Predecessors for each op (dependency graph)
  double cost;
  
  HardwarePattern(int64_t i, const std::string& n, int64_t f);
};

//===----------------------------------------------------------------------===//
// Functional Unit (FU) - A single hardware execution unit
//===----------------------------------------------------------------------===//
//
// A Functional Unit represents a single hardware unit that can execute exactly
// one type of operation (e.g., adder, multiplier, load unit).
//
// Key Properties:
// ---------------
// 1. SINGLE OPERATION TYPE: Each FU executes exactly one operation type.
//    For example, an "adder" FU only executes neura.add operations.
//
// 2. MULTIPLE INSTANCES: A template can have multiple FUs of the same type.
//    For example, two adders to support patterns needing parallel additions.
//
// 3. DIRECT CONNECTIONS: FUs are connected directly to each other, forming
//    a dataflow graph within the template.
//
// Example:
// --------
// Consider a template supporting pattern: gep -> load -> add -> store
//
// Template structure:
//   ┌─────┐     ┌──────┐     ┌─────┐     ┌───────┐
//   │ gep │ --> │ load │ --> │ add │ --> │ store │
//   │FU 0 │     │ FU 1 │     │FU 2 │     │ FU 3  │
//   └─────┘     └──────┘     └─────┘     └───────┘
//
// For patterns with parallel operations (e.g., add + mul -> store):
//   ┌─────┐
//   │ add │ ──┐
//   │FU 0 │   │     ┌───────┐
//   └─────┘   ├──-> │ store │
//   ┌─────┐   │     │ FU 2  │
//   │ mul │ ──┘     └───────┘
//   │FU 1 │
//   └─────┘
//
//===----------------------------------------------------------------------===//
struct Functional Unit {
  int id;                    // Unique FU ID within the template
  std::string op_type;       // Operation type this FU executes (e.g., "neura.add")
  
  Functional Unit(int i, const std::string& op);
};

// Execution stage for a pattern - contains FU indices that can execute in parallel.
struct ExecutionStage {
  std::vector<int> fus;          // FUs that execute in this stage (parallel)
  std::vector<std::string> ops;  // Corresponding operations
};

// Execution plan for a pattern on a hardware template.
struct PatternExecutionPlan {
  int64_t pattern_id;
  std::string pattern_name;
  std::vector<ExecutionStage> stages;  // Ordered stages of execution
};

// Operations supported by a hardware template.
struct TemplateSupportedOps {
  int template_id;
  std::set<std::string> single_ops;    // Individual ops this template can support
  std::vector<int64_t> composite_ops;  // Pattern IDs (composite operations)
};

class OperationCostModel {
public:
  OperationCostModel();
  double get(const std::string& op) const;
  double fu_cost(const std::string& op) const;
  double pattern_cost(const std::vector<std::string>& ops) const;
private:
  std::map<std::string, double> costs;
};

//===----------------------------------------------------------------------===//
// HardwareTemplate - A collection of FUs forming a reusable hardware block
//===----------------------------------------------------------------------===//
//
// A HardwareTemplate contains multiple Functional Units connected together.
// Multiple patterns can be mapped to the same template by reusing FUs.
//
// Key differences from the old slot-based design:
// - Each FU has exactly one operation type (no multiplexing within FU)
// - Template can have multiple FUs of the same type
// - Connections are between specific FU IDs, not abstract slot positions
//
//===----------------------------------------------------------------------===//
struct HardwareTemplate {
  int id;
  std::vector<Functional Unit> fus;             // All FUs in this template
  std::vector<int64_t> patterns;                // Pattern IDs mapped to this template
  std::map<int64_t, std::vector<int>> mapping;  // pattern_id -> FU id sequence
  std::set<std::pair<int, int>> connections;    // FU connections: (from_fu_id, to_fu_id)
  int instances;
  
  HardwareTemplate(int i);
  
  // Adds a new FU with the given operation type, returns its ID.
  int add_fu(const std::string& op_type);
  
  // Finds an existing FU that can handle the operation, or -1 if none available.
  int find_available_fu(const std::string& op_type, const std::set<int>& used_fus) const;
  
  // Finds a mapping for a pattern into the existing template.
  // Returns FU IDs for each operation, or empty if no valid mapping exists.
  std::vector<int> find_mapping(const HardwarePattern& pat) const;
  
  // Tries to accommodate a pattern, potentially adding new FUs.
  // Returns true if successful, with the mapping and cost increase.
  bool try_accommodate(const HardwarePattern& pat, const OperationCostModel& cm, 
                       std::vector<int>& out_mapping, double& out_cost_increase);
  
  // Applies a mapping to the template.
  void apply_mapping(const HardwarePattern& pat, const std::vector<int>& m);
  
  // Computes the total cost of the template.
  double compute_cost(const OperationCostModel& cm) const;
  
  // Checks if two operations are compatible (can potentially share resources in future).
  static bool compatible(const std::string& a, const std::string& b);
  
private:
  // DFS helper for finding mappings.
  void dfs_find_mapping(const HardwarePattern& pat, size_t op_idx, std::vector<int>& cur_mapping, std::set<int>& used_fus, std::vector<int>& best_mapping, int& best_reuse_count) const;
};

// Extracts all patterns from module.
void extract_patterns(ModuleOp module, std::vector<HardwarePattern>& patterns, OperationCostModel& cost_model);

// Extracts all standalone operations from module (ops not inside FusedOp).
void extract_all_standalone_ops(ModuleOp module, std::set<std::string>& all_ops);

// Creates hardware templates from patterns.
void create_hardware_templates(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates, OperationCostModel& cost_model);

// Generates FU connections for all templates based on pattern dependencies.
void generate_connections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates);

// Generates optimized FU connections (removes redundant connections using transitive reachability).
void generate_optimized_connections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates);

// Generates execution plans for all patterns on their assigned templates.
void generate_execution_plans(const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, std::vector<PatternExecutionPlan>& plans);

// Collects supported operations (single + composite) for each template.
void collect_supported_operations(const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const std::set<std::string>& all_dfg_ops, std::vector<TemplateSupportedOps>& supported_ops);

// Calculates total cost of templates.
double calculate_total_cost(const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model);

// Writes hardware configuration to JSON file (extended version with execution plans and supported ops).
void write_hardware_config_json(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model, const std::vector<PatternExecutionPlan>& execution_plans, const std::vector<TemplateSupportedOps>& supported_ops);

// Legacy version for backward compatibility.
void write_hardware_config_json(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model);

} // namespace mlir::neura

#endif // NEURA_DIALECT_TRANSFORMS_GRAPHMINING_HARDWARETEMPLATE_H

