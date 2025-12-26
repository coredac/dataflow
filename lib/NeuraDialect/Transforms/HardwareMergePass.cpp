//===- HardwareMergePass.cpp - Aggressive Hardware Template Merging -------===//
//
// This pass maximizes pattern coverage with minimum hardware cost.
// It supports slot bypassing, dynamic connections, template extension,
// and cost-based pattern processing.
//
//===----------------------------------------------------------------------===//

#include "NeuraDialect/Transforms/GraphMining/HardwareTemplate.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

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
    
    extractPatterns(getOperation(), patterns, costModel);
    createHardwareTemplates(patterns, templates, costModel);
    writeHardwareConfigJson(outputFile.getValue(), patterns, templates, costModel);
    
    double totalCost = calculateTotalCost(templates, costModel);
    llvm::errs() << "\nResult: " << templates.size() << " templates, "
                 << "cost: " << totalCost << "\n";
    llvm::errs() << "========================================\n\n";
  }
};

} // namespace

namespace mlir::neura {
std::unique_ptr<Pass> createHardwareMergePass() {
  return std::make_unique<HardwareMergePass>();
}
} // namespace mlir::neura
