// TaskflowPasses.h - Header file for Taskflow passes

#ifndef TASKFLOW_PASSES_H
#define TASKFLOW_PASSES_H

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>
namespace mlir {
namespace taskflow {

void registerTaskflowConversionPassPipeline();
void registerTosaToAffineConversionPassPipeline();

// Passes defined in TaskflowPasses.td
#define GEN_PASS_DECL
#include "TaskflowDialect/TaskflowPasses.h.inc"
std::unique_ptr<mlir::Pass> createConstructHyperblockFromTaskPass();
std::unique_ptr<mlir::Pass> createClassifyCountersPass();
std::unique_ptr<mlir::Pass> createAllocateCgraToTaskPass();

// Runs the CGRA task placement logic directly on a function.
// grid_rows/grid_cols default to 4x4 (kCgraGridRows/kCgraGridCols).
void runAllocateCgraToTask(mlir::func::FuncOp func,
                      int grid_rows = 4, int grid_cols = 4);

//=========================================================//
// Optimization Passes
//=========================================================//
std::unique_ptr<mlir::Pass> createAffineLoopTreeSerializationPass();
std::unique_ptr<mlir::Pass> createAffineLoopPerfectionPass();
std::unique_ptr<mlir::Pass> createMemoryAccessStreamingFusionPass();
std::unique_ptr<mlir::Pass> createResourceAwareTaskOptimizationPass();

#define GEN_PASS_REGISTRATION
#include "TaskflowDialect/TaskflowPasses.h.inc"
} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_PASSES_H