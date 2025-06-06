// ConversionPasses.h - Header file for conversion passes

#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {

// Passes defined in GraphPasses.td.
#define GEN_PASS_DECL
#include "Conversion/ConversionPasses.h.inc"

// Conversion passes.
std::unique_ptr<mlir::Pass> createLowerArithToNeuraPass();
std::unique_ptr<mlir::Pass> createLowerLlvmToNeuraPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/ConversionPasses.h.inc"

} // namespace mlir

#endif // CONVERSION_PASSES_H