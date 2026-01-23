//===- HardwareTemplate.h - Hardware Template Data Structures and Helpers -===//
//
// This file contains declarations for hardware template data structures and
// helper functions for hardware template merging.
//
// The hardware template system maximizes pattern coverage while minimizing
// hardware cost through resource sharing. Key concepts:
//
// - HardwareSlot: A single FU position where ops execute mutually exclusively
// - HardwareTemplate: A pipeline of slots supporting multiple patterns
// - HardwarePattern: A sequence of operations mapped to template slots
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
// HardwareSlot - A configurable execution unit within a Hardware Template
//===----------------------------------------------------------------------===//
//
// A HardwareSlot represents a single pipeline stage or functional unit (FU) position
// within a hardware template. Each slot can support multiple compatible operations
// through hardware multiplexing.
//
// Key Properties:
// ---------------
// 1. MUTUAL EXCLUSION: Operations within the same slot CANNOT execute simultaneously.
//    Only ONE operation can be active in a slot at any given time. This is because
//    a slot represents a single physical FU that is configured at runtime.
//
// 2. OPERATION SHARING: Multiple compatible ops can share the same slot to reduce
//    hardware cost. For example, `neura.add` and `neura.sub` can share an ALU slot.
//
// 3. PIPELINE POSITION: Slots are ordered (slot 0 -> slot 1 -> slot 2 ...) and data
//    flows forward through connections. A slot cannot send data backwards.
//
// When to Use Slots:
// ------------------
// - Resource sharing: When multiple patterns use similar operations (e.g., add/sub),
//   mapping them to the same slot saves hardware by sharing the underlying FU.
// - Pipeline stages: Each slot acts as a stage in a pipelined datapath, allowing
//   multiple patterns to be executed with different timing.
// - Bypass support: Connections can skip slots (e.g., 0->2) for patterns that
//   don't use intermediate slots.
//
// Example:
// --------
// Consider a template supporting two patterns:
//   Pattern A: gep -> load (slots [0, 1])
//   Pattern B: phi_start -> gep -> load (slots [0, 1, 2])
//
// Template structure:
//   ┌─────────┐     ┌─────────┐     ┌─────────┐
//   │ Slot 0  │ --> │ Slot 1  │ --> │ Slot 2  │
//   │phi_start│     │  gep    │     │  load   │
//   └─────────┘     └─────────┘     └─────────┘
//
// For Pattern A (gep->load):
//   - Uses slots [1, 2], bypassing slot 0
//   - Stage 0: Slot 1 executes gep
//   - Stage 1: Slot 2 executes load
//
// For Pattern B (phi_start->gep->load):
//   - Uses slots [0, 1, 2]
//   - Stage 0: Slot 0 executes phi_start
//   - Stage 1: Slot 1 executes gep
//   - Stage 2: Slot 2 executes load
//
// Slot Sharing Example:
// --------------------
// If two patterns need:
//   Pattern X: add -> mul
//   Pattern Y: sub -> mul
//
// Since add and sub are compatible (same ALU), they can share slot 0:
//   ┌─────────────┐     ┌─────────┐
//   │   Slot 0    │ --> │ Slot 1  │
//   │ add OR sub  │     │   mul   │
//   └─────────────┘     └─────────┘
//
// At runtime, slot 0's FU is configured for either add or sub based on which
// pattern is executing. Both patterns use the same physical hardware.
//
//===----------------------------------------------------------------------===//
struct HardwareSlot {
  int id;                      // Slot position in the template pipeline (0, 1, 2, ...)
  std::set<std::string> ops;   // Set of operations this slot can execute (mutually exclusive)
  
  HardwareSlot(int i);
};

// Execution stage for a pattern - contains slot indices that can execute in parallel.
struct ExecutionStage {
  std::vector<int> slots;  // Slots that execute in this stage (parallel)
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
  std::set<std::string> single_ops;  // Individual ops this template can support
  std::vector<int64_t> composite_ops;  // Pattern IDs (composite operations)
};

class OperationCostModel {
public:
  OperationCostModel();
  double get(const std::string& op) const;
  double slot_cost(const std::set<std::string>& ops) const;
  double pattern_cost(const std::vector<std::string>& ops) const;
private:
  std::map<std::string, double> costs;
};

struct HardwareTemplate {
  int id;
  std::vector<HardwareSlot> slots;
  std::vector<int64_t> patterns;
  std::map<int64_t, std::vector<int>> mapping;
  std::set<std::pair<int, int>> connections;  // Slot connections: (from_slot, to_slot)
  int instances;
  
  HardwareTemplate(int i);
  void add_slot();
  void insert_slot_at_front();
  bool can_route(int from, int to) const;
  std::vector<int> find_mapping(const std::vector<std::string>& pat_ops) const;
  bool slot_can_handle(size_t s, const std::string& op) const;
  static bool compatible(const std::string& a, const std::string& b);
  bool try_accommodate(const HardwarePattern& pat, const OperationCostModel& cm, std::vector<int>& out_mapping, double& out_cost_increase);
  void apply_mapping(const HardwarePattern& pat, const std::vector<int>& m);
  double compute_cost(const OperationCostModel& cm) const;
  
private:
  void dfs_with_scoring(const std::vector<std::string>& pat_ops, size_t op_idx, int prev_slot, std::vector<int> cur, std::vector<int>& best_mapping, int& best_score) const;
  std::vector<int> dfs(const std::vector<std::string>& pat_ops, size_t op_idx, int prev_slot, std::vector<int> cur) const;
};

// Extracts all patterns from module.
void extract_patterns(ModuleOp module, std::vector<HardwarePattern>& patterns, OperationCostModel& cost_model);

// Extracts all standalone operations from module (ops not inside FusedOp).
void extract_all_standalone_ops(ModuleOp module, std::set<std::string>& all_ops);

// Creates hardware templates from patterns.
void create_hardware_templates(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates, OperationCostModel& cost_model);

// Generates optimized slot connections for all templates.
// Connections are minimized using transitive reachability (bypass support).
// Only adds connection A->C if there's no path A->B->C already.
void generate_optimized_connections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates);

// Generates slot connections for all templates based on pattern mappings.
void generate_connections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates);

// Generates execution plans for all patterns on their assigned templates.
// Returns the mapping from (template_id, pattern_id) to execution plan.
void generate_execution_plans(const std::vector<HardwarePattern>& patterns, 
                            const std::vector<HardwareTemplate>& templates,
                            std::vector<PatternExecutionPlan>& plans);

// Collects supported operations (single + composite) for each template.
void collect_supported_operations(const std::vector<HardwarePattern>& patterns,
                                const std::vector<HardwareTemplate>& templates,
                                const std::set<std::string>& all_dfg_ops,
                                std::vector<TemplateSupportedOps>& supported_ops);

// Calculates total cost of templates.
double calculate_total_cost(const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model);

// Writes hardware configuration to JSON file (extended version with execution plans and supported ops).
void write_hardware_config_json(const std::string& path, 
                             const std::vector<HardwarePattern>& patterns, 
                             const std::vector<HardwareTemplate>& templates, 
                             const OperationCostModel& cost_model,
                             const std::vector<PatternExecutionPlan>& execution_plans,
                             const std::vector<TemplateSupportedOps>& supported_ops);

// Legacy version for backward compatibility.
void write_hardware_config_json(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& cost_model);

} // namespace mlir::neura

#endif // NEURA_DIALECT_TRANSFORMS_GRAPHMINING_HARDWARETEMPLATE_H

