//===- AssignTaskTargetPass.cpp - Assign hardware targets to tasks --------===//
//
// This pass assigns hardware target attributes to compute tasks (functions)
// based on task names. It helps partition the workload across different
// hardware units (CPU, CGRA, DOE, etc.) in heterogeneous computing systems.
//
//===----------------------------------------------------------------------===//

#include "Conversion/ConversionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Determines the hardware target for a given function based on its name.
/// This function implements a simple pattern-matching strategy:
///   - ray_sampler* -> CPU
///   - hash_encoder* -> DOE
///   - nerf_mlp* -> CGRA
///   - nerf_forward (top-level) -> CPU
///   - default -> CPU
static StringRef matchHardwareTarget(StringRef funcName) {
  // Top-level function: runs on CPU as coordinator
  if (funcName == "nerf_forward") {
    return "cpu";
  }

  // Pattern matching for compute tasks
  if (funcName.contains("ray_sampler") || funcName.contains("sampler")) {
    return "cpu";
  }

  if (funcName.contains("hash_encoder") || funcName.contains("encoder")) {
    return "doe";
  }

  if (funcName.contains("nerf_mlp") || funcName.contains("mlp")) {
    return "cgra";
  }

  // Default target
  return "cpu";
}

//===----------------------------------------------------------------------===//
// AssignTaskTarget Pass
//===----------------------------------------------------------------------===//

struct AssignTaskTargetPass
    : public PassWrapper<AssignTaskTargetPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignTaskTargetPass)

  StringRef getArgument() const final { return "assign-task-target"; }

  StringRef getDescription() const final {
    return "Assign hardware targets to compute tasks (functions) based on "
           "task names";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    // Statistics
    unsigned totalFuncs = 0;
    unsigned assignedFuncs = 0;
    llvm::DenseMap<StringRef, unsigned> targetStats;

    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "AssignTaskTarget Pass\n";
    llvm::errs() << "========================================\n\n";

    // Walk through all functions in the module
    module.walk([&](func::FuncOp funcOp) {
      totalFuncs++;
      StringRef funcName = funcOp.getName();

      // Determine hardware target based on function name
      StringRef target = matchHardwareTarget(funcName);

      // Set the target.device attribute
      funcOp->setAttr("target.device", builder.getStringAttr(target));

      assignedFuncs++;
      targetStats[target]++;

      llvm::errs() << "  [ASSIGN] " << funcName << " -> " << target << "\n";
    });

    // Print summary
    llvm::errs() << "\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "Summary\n";
    llvm::errs() << "========================================\n";
    llvm::errs() << "Total functions:    " << totalFuncs << "\n";
    llvm::errs() << "Assigned functions: " << assignedFuncs << "\n";

    if (!targetStats.empty()) {
      llvm::errs() << "\nTarget distribution:\n";
      for (auto &entry : targetStats) {
        llvm::errs() << "  " << entry.first << ": " << entry.second
                     << " function(s)\n";
      }
    }

    llvm::errs() << "========================================\n\n";
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {

std::unique_ptr<Pass> createAssignTaskTargetPass() {
  return std::make_unique<AssignTaskTargetPass>();
}

}  // namespace mlir
