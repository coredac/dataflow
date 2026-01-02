//===- HardwareMergePass.cpp - Aggressive Hardware Template Merging -------===//
//
// This pass maximizes pattern coverage with minimum hardware cost.
// It supports slot bypassing, dynamic connections, template extension,
// and cost-based pattern processing.
//
// Features:
// 1. Optimized slot connections with bypass support (transitive reachability).
// 2. Coverage of all operators including standalone (non-fused) basic operators.
// 3. Execution plans for each pattern showing parallel execution stages.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Transforms/GraphMining/HardwareTemplate.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
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
    return "Maximally merge hardware templates with bypass support.";
  }
  
  Option<std::string> outputFile{*this, "output", 
    llvm::cl::desc("Output JSON"), llvm::cl::init("hardware_config.json")};
  
  void runOnOperation() override {
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "HardwareMergePass: Aggressive Merging\n";
    llvm::errs() << "========================================\n";
    
    std::vector<HardwarePattern> patterns;
    std::vector<HardwareTemplate> templates;
    OperationCostModel costModel;
    
    // Step 1: Extract all patterns from fused ops.
    extractPatterns(getOperation(), patterns, costModel);
    llvm::errs() << "[Step 1] Extracted " << patterns.size() << " patterns\n";
    
    // Step 2: Extract all standalone operations (not inside fused ops).
    // These are basic operators that need to be covered by hardware templates.
    std::set<std::string> allStandaloneOps;
    extractAllStandaloneOps(getOperation(), allStandaloneOps);
    llvm::errs() << "[Step 2] Found " << allStandaloneOps.size() << " standalone ops in DFG\n";
    for (const auto& op : allStandaloneOps) {
      llvm::errs() << "  - " << op << "\n";
    }
    
    // Step 3: Create hardware templates from patterns.
    createHardwareTemplates(patterns, templates, costModel);
    llvm::errs() << "[Step 3] Created " << templates.size() << " hardware templates\n";
    
    // Step 4: Generate optimized slot connections with bypass support.
    llvm::errs() << "[Step 4] Generating optimized slot connections:\n";
    generateOptimizedConnections(patterns, templates);
    
    // Step 5: Generate execution plans for each pattern.
    std::vector<PatternExecutionPlan> executionPlans;
    generateExecutionPlans(patterns, templates, executionPlans);
    llvm::errs() << "[Step 5] Generated " << executionPlans.size() << " execution plans\n";
    
    // Step 6: Collect supported operations for each template.
    // Combines all ops from patterns + standalone ops from DFG.
    std::set<std::string> allDfgOps = allStandaloneOps;
    for (const auto& p : patterns) {
      for (const auto& op : p.ops) {
        allDfgOps.insert(op);
      }
    }
    
    std::vector<TemplateSupportedOps> supportedOps;
    collectSupportedOperations(patterns, templates, allDfgOps, supportedOps);
    llvm::errs() << "[Step 6] Collected supported operations for each template\n";
    
    // Step 7: Output results.
    writeHardwareConfigJson(outputFile.getValue(), patterns, templates, costModel, 
                            executionPlans, supportedOps);
    
    // Print summary.
    double totalCost = calculateTotalCost(templates, costModel);
    llvm::errs() << "\n========================================\n";
    llvm::errs() << "Summary:\n";
    llvm::errs() << "  - Templates: " << templates.size() << "\n";
    llvm::errs() << "  - Total cost: " << totalCost << "\n";
    
    // Print template details.
    for (const auto& tmpl : templates) {
      llvm::errs() << "\n  Template " << tmpl.id << ":\n";
      llvm::errs() << "    Slots: " << tmpl.slots.size() << "\n";
      llvm::errs() << "    Connections: ";
      for (const auto& conn : tmpl.connections) {
        llvm::errs() << "[" << conn.first << "->" << conn.second << "] ";
      }
      llvm::errs() << "\n";
      
      // Find supported ops for this template.
      for (const auto& sop : supportedOps) {
        if (sop.templateId == tmpl.id) {
          llvm::errs() << "    Single ops: ";
          for (const auto& op : sop.singleOps) {
            llvm::errs() << op << " ";
          }
          llvm::errs() << "\n";
          llvm::errs() << "    Composite ops (patterns): ";
          for (int64_t pid : sop.compositeOps) {
            llvm::errs() << "P" << pid << " ";
          }
          llvm::errs() << "\n";
          break;
        }
      }
    }
    
    // Print execution plans.
    llvm::errs() << "\n  Execution Plans:\n";
    for (const auto& plan : executionPlans) {
      llvm::errs() << "    Pattern " << plan.patternId << " (" << plan.patternName << "):\n";
      for (size_t i = 0; i < plan.stages.size(); ++i) {
        const auto& stage = plan.stages[i];
        llvm::errs() << "      Stage " << i << ": slots[";
        for (size_t j = 0; j < stage.slots.size(); ++j) {
          if (j) llvm::errs() << ",";
          llvm::errs() << stage.slots[j];
        }
        llvm::errs() << "] ops[";
        for (size_t j = 0; j < stage.ops.size(); ++j) {
          if (j) llvm::errs() << ",";
          llvm::errs() << stage.ops[j];
        }
        llvm::errs() << "]\n";
      }
    }
    
    llvm::errs() << "========================================\n\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createHardwareMergePass() {
  return std::make_unique<HardwareMergePass>();
}
} // namespace mlir::neura
