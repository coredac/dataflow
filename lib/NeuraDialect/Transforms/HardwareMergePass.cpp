//===- HardwareMergePass.cpp - Hardware Template Merging Pass -------------===//
//
// This pass maximizes pattern coverage with minimum hardware cost by merging
// patterns into shared hardware templates. It supports slot bypassing,
// optimized connections with transitive reachability, and generates execution
// plans for parallel execution stages.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Transforms/GraphMining/HardwareTemplate.h"
#include "mlir/Pass/Pass.h"
#include <vector>
#include <set>

using namespace mlir;
using namespace mlir::neura;

#define GEN_PASS_DEF_HARDWAREMERGE
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {

struct HardwareMergePass
    : public PassWrapper<HardwareMergePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HardwareMergePass)
  
  HardwareMergePass() = default;
  HardwareMergePass(const HardwareMergePass &pass)
      : PassWrapper<HardwareMergePass, OperationPass<ModuleOp>>(pass) {}
  
  StringRef getArgument() const override { return "hardware-merge"; }
  StringRef getDescription() const override {
    return "Merges hardware templates with bypass support.";
  }
  
  Option<std::string> outputFile{*this, "output", 
    llvm::cl::desc("Output JSON file path"), llvm::cl::init("hardware_config.json")};
  
  // Runs the hardware merge pass on the module.
  void runOnOperation() override {
    std::vector<HardwarePattern> patterns;
    std::vector<HardwareTemplate> templates;
    OperationCostModel cost_model;
    
    extract_patterns(getOperation(), patterns, cost_model);
    
    std::set<std::string> all_standalone_ops;
    extract_all_standalone_ops(getOperation(), all_standalone_ops);
    
    create_hardware_templates(patterns, templates, cost_model);
    
    generate_optimized_connections(patterns, templates);
    
    std::vector<PatternExecutionPlan> execution_plans;
    generate_execution_plans(patterns, templates, execution_plans);
    
    std::set<std::string> all_dfg_ops = all_standalone_ops;
    for (const auto& p : patterns) {
      for (const auto& op : p.ops) {
        all_dfg_ops.insert(op);
      }
    }
    
    std::vector<TemplateSupportedOps> supported_ops;
    collect_supported_operations(patterns, templates, all_dfg_ops, supported_ops);
    
    write_hardware_config_json(outputFile.getValue(), patterns, templates, cost_model, execution_plans, supported_ops);
  }
};

} // namespace

namespace mlir::neura {
// Creates an instance of the hardware merge pass.
std::unique_ptr<Pass> createHardwareMergePass() {
  return std::make_unique<HardwareMergePass>();
}
} // namespace mlir::neura
