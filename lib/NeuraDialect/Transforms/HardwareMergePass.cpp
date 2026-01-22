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
    OperationCostModel costModel;
    
    extractPatterns(getOperation(), patterns, costModel);
    
    std::set<std::string> allStandaloneOps;
    extractAllStandaloneOps(getOperation(), allStandaloneOps);
    
    createHardwareTemplates(patterns, templates, costModel);
    
    generateOptimizedConnections(patterns, templates);
    
    std::vector<PatternExecutionPlan> executionPlans;
    generateExecutionPlans(patterns, templates, executionPlans);
    
    std::set<std::string> allDfgOps = allStandaloneOps;
    for (const auto& p : patterns) {
      for (const auto& op : p.ops) {
        allDfgOps.insert(op);
      }
    }
    
    std::vector<TemplateSupportedOps> supportedOps;
    collectSupportedOperations(patterns, templates, allDfgOps, supportedOps);
    
    writeHardwareConfigJson(outputFile.getValue(), patterns, templates, costModel, executionPlans, supportedOps);
  }
};

} // namespace

namespace mlir::neura {
// Creates an instance of the hardware merge pass.
std::unique_ptr<Pass> createHardwareMergePass() {
  return std::make_unique<HardwareMergePass>();
}
} // namespace mlir::neura
