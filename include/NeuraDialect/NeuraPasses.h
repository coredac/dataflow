// NeuraPasses.h - Header file for Neura passes

#ifndef NEURA_PASSES_H
#define NEURA_PASSES_H

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
namespace neura {

// Passes defined in GraphPasses.td
#define GEN_PASS_DECL
#include "NeuraDialect/NeuraPasses.h.inc"
std::unique_ptr<mlir::Pass> createInsertMovPass();
std::unique_ptr<mlir::Pass> createFusePatternsPass();
std::unique_ptr<mlir::Pass> createAssignAcceleratorPass();

#define GEN_PASS_REGISTRATION
#include "NeuraDialect/NeuraPasses.h.inc"

} // namespace neura
} // namespace mlir

#endif // NEURA_PASSES_H