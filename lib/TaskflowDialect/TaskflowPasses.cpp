#include "TaskflowDialect/TaskflowPasses.h"
#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

// This pass pipeline can convert all the other dialects into the Neura dialect.
void mlir::taskflow::registerTaskflowConversionPassPipeline() {
  PassPipelineRegistration<>(
      "taskflow-conversion",
      "Convertes affine dialects to taskflow dialect with neura.kernel ops.",
      [](OpPassManager &pm) {
        pm.addPass(mlir::createConvertAffineToTaskflowPass());
        pm.addPass(mlir::taskflow::createConstructHyperblockFromTaskPass());
        pm.addPass(mlir::taskflow::createClassifyCountersPass());
        pm.addPass(mlir::createConvertTaskflowToNeuraPass());
      });
}

// This pass pipeline converts TOSA dialect to Affine dialect with cleanup.
void mlir::taskflow::registerTosaToAffineConversionPassPipeline() {
  PassPipelineRegistration<>(
      "tosa-to-affine-conversion",
      "Complete pipeline: TOSA to Linalg to Affine with subview/copy cleanup.",
      [](OpPassManager &pm) {
        // Step 1: TOSA to Linalg Named operations.
        pm.addPass(mlir::tosa::createTosaInferShapesPass());
        pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
        pm.addPass(mlir::tosa::createTosaToLinalgNamed());

        // Step 2: TOSA to Linalg (remaining ops).
        pm.addPass(mlir::tosa::createTosaToLinalg());

        // Step 3: TOSA to Standard.
        pm.addPass(mlir::tosa::createTosaToArith());
        pm.addPass(mlir::tosa::createTosaToTensor());

        // Step 4: Linalg Generalization.
        pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());

        // Step 5: Canonicalization.
        pm.addPass(mlir::createCanonicalizerPass());

        // Step 6: Bufferization with proper options.
        bufferization::OneShotBufferizationOptions bufferizationOptions;
        bufferizationOptions.bufferizeFunctionBoundaries = true;
        bufferizationOptions.setFunctionBoundaryTypeConversion(
            bufferization::LayoutMapOption::IdentityLayoutMap);
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(
            bufferizationOptions));

        // Step 7: Linalg to Affine Loops.
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

        // Step 8: Cleanup subview and copy operations.
        pm.addPass(mlir::createFoldSubViewPass());
        pm.addPass(mlir::createConvertCopyToAffineLoopsPass());

        // Step 9: Final Affine cleanup.
        pm.addPass(mlir::affine::createAffineLoopNormalizePass());
        pm.addPass(mlir::affine::createSimplifyAffineStructuresPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}