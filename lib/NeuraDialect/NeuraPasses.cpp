#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"

// This pass pipeline can convert all the other dialects into the Neura dialect
void mlir::neura::registerNeuraConversionPassPipeline() {
  PassPipelineRegistration<>(
      "neura-conversion", "Convert all dialects to Neura dialect",
      [](OpPassManager &pm) {
        pm.addPass(mlir::neura::createAssignAcceleratorPass());
        // Convert all the other dialects into the Neura dialect
        pm.addPass(mlir::createLowerArithToNeuraPass());
        pm.addPass(mlir::createLowerLlvmToNeuraPass());
      });
}