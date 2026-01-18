// TaskflowPasses.h - Header file for Taskflow passes.

#ifndef TASKFLOW_PASSES_H
#define TASKFLOW_PASSES_H

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace mlir {
namespace taskflow {

// Passes defined in TaskflowPasses.td.
#define GEN_PASS_DECL
#include "TaskflowDialect/TaskflowPasses.h.inc"

/// Creates a pass that constructs hyperblocks and counter chains from tasks.
std::unique_ptr<mlir::Pass> createConstructHyperblockFromTaskPass();

/// Creates a pass that optimizes the task graph by fusing hyperblocks and tasks.
std::unique_ptr<mlir::Pass> createOptimizeTaskGraphPass();

#define GEN_PASS_REGISTRATION
#include "TaskflowDialect/TaskflowPasses.h.inc"

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_PASSES_H