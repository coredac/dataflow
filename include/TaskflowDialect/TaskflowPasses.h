// TaskflowPasses.h - Header file for Taskflow passes

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
// Passes defined in TaskflowPasses.td
#define GEN_PASS_DECL
#include "TaskflowDialect/TaskflowPasses.h.inc"
std::unique_ptr<mlir::Pass> createAffineLoopTreeSerializationPass();
std::unique_ptr<mlir::Pass> createConstructHyperblockFromTaskPass();
std::unique_ptr<mlir::Pass> createClassifyCountersPass();
std::unique_ptr<mlir::Pass> createCanonicalizeTaskPass();

#define GEN_PASS_REGISTRATION
#include "TaskflowDialect/TaskflowPasses.h.inc"
} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_PASSES_H