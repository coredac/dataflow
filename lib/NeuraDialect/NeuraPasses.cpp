#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraTypes.h"
#include "Conversion/ConversionPasses.h"

// This pass pipeline can convert all the other dialects into the Neura dialect
void mlir::neura::registerNeuraLegalizePassPipeline() {
  PassPipelineRegistration<>("neura-legalize",
                             "Legalize operations to Neura dialect",
                             [](OpPassManager &pm) {
                                // Convert all the other dialects into the Neura dialect
                                pm.addPass(mlir::createLowerAffineToNeuraPass());
                                pm.addPass(mlir::createLowerArithToNeuraPass());
                                pm.addPass(mlir::createLowerLlvmToNeuraPass());

                                // Insert data and control movement operations
                                // pm.addPass(mlir::neura::createLeveragePredicatedValuePass());
                                // pm.addPass(mlir::neura::createInsertDataMovPass());
                                // pm.addPass(mlir::neura::createInsertCtrlMovPass());
                                // pm.addPass(mlir::neura::createTransformCtrlToDataFlowPass());
                             });
}