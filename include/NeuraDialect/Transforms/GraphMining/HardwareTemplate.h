//===- HardwareTemplate.h - Hardware Template Data Structures and Helpers -===//
//
// This file contains declarations for hardware template data structures and
// helper functions for hardware template merging.
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
  std::vector<int> opLevels;  // Topological level for each op (ops at same level can run in parallel)
  std::vector<std::vector<int>> opPreds;  // Predecessors for each op (dependency graph)
  double cost;
  
  HardwarePattern(int64_t i, const std::string& n, int64_t f);
};

struct HardwareSlot {
  int id;
  std::set<std::string> ops;
  HardwareSlot(int i);
};

// Execution stage for a pattern - contains slot indices that can execute in parallel.
struct ExecutionStage {
  std::vector<int> slots;  // Slots that execute in this stage (parallel)
  std::vector<std::string> ops;  // Corresponding operations
};

// Execution plan for a pattern on a hardware template.
struct PatternExecutionPlan {
  int64_t patternId;
  std::string patternName;
  std::vector<ExecutionStage> stages;  // Ordered stages of execution
};

// Operations supported by a hardware template.
struct TemplateSupportedOps {
  int templateId;
  std::set<std::string> singleOps;  // Individual ops this template can support
  std::vector<int64_t> compositeOps;  // Pattern IDs (composite operations)
};

class OperationCostModel {
public:
  OperationCostModel();
  double get(const std::string& op) const;
  double slotCost(const std::set<std::string>& ops) const;
  double patternCost(const std::vector<std::string>& ops) const;
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
  void addSlot();
  void insertSlotAtFront();
  bool canRoute(int from, int to) const;
  std::vector<int> findMapping(const std::vector<std::string>& patOps) const;
  bool slotCanHandle(size_t s, const std::string& op) const;
  static bool compatible(const std::string& a, const std::string& b);
  bool tryAccommodate(const HardwarePattern& pat, const OperationCostModel& cm, std::vector<int>& outMapping, double& outCostIncrease);
  void applyMapping(const HardwarePattern& pat, const std::vector<int>& m);
  double computeCost(const OperationCostModel& cm) const;
  
private:
  void dfsWithScoring(const std::vector<std::string>& patOps, size_t opIdx, int prevSlot, std::vector<int> cur, std::vector<int>& bestMapping, int& bestScore) const;
  std::vector<int> dfs(const std::vector<std::string>& patOps, size_t opIdx, int prevSlot, std::vector<int> cur) const;
};

// Extracts all patterns from module.
void extractPatterns(ModuleOp module, std::vector<HardwarePattern>& patterns, OperationCostModel& costModel);

// Extracts all standalone operations from module (ops not inside FusedOp).
void extractAllStandaloneOps(ModuleOp module, std::set<std::string>& allOps);

// Creates hardware templates from patterns.
void createHardwareTemplates(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates, OperationCostModel& costModel);

// Generates optimized slot connections for all templates.
// Connections are minimized using transitive reachability (bypass support).
// Only adds connection A->C if there's no path A->B->C already.
void generateOptimizedConnections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates);

// Generates slot connections for all templates based on pattern mappings.
void generateConnections(const std::vector<HardwarePattern>& patterns, std::vector<HardwareTemplate>& templates);

// Generates execution plans for all patterns on their assigned templates.
// Returns the mapping from (template_id, pattern_id) to execution plan.
void generateExecutionPlans(const std::vector<HardwarePattern>& patterns, 
                            const std::vector<HardwareTemplate>& templates,
                            std::vector<PatternExecutionPlan>& plans);

// Collects supported operations (single + composite) for each template.
void collectSupportedOperations(const std::vector<HardwarePattern>& patterns,
                                const std::vector<HardwareTemplate>& templates,
                                const std::set<std::string>& allDfgOps,
                                std::vector<TemplateSupportedOps>& supportedOps);

// Calculates total cost of templates.
double calculateTotalCost(const std::vector<HardwareTemplate>& templates, const OperationCostModel& costModel);

// Writes hardware configuration to JSON file (extended version with execution plans and supported ops).
void writeHardwareConfigJson(const std::string& path, 
                             const std::vector<HardwarePattern>& patterns, 
                             const std::vector<HardwareTemplate>& templates, 
                             const OperationCostModel& costModel,
                             const std::vector<PatternExecutionPlan>& executionPlans,
                             const std::vector<TemplateSupportedOps>& supportedOps);

// Legacy version for backward compatibility.
void writeHardwareConfigJson(const std::string& path, const std::vector<HardwarePattern>& patterns, const std::vector<HardwareTemplate>& templates, const OperationCostModel& costModel);

} // namespace mlir::neura

#endif // NEURA_DIALECT_TRANSFORMS_GRAPHMINING_HARDWARETEMPLATE_H

