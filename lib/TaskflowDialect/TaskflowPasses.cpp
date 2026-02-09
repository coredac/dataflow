#include "TaskflowDialect/TaskflowPasses.h"
#include "Conversion/ConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
      "Complete pipeline: TOSA to Linalg to Affine with subview/copy cleanup",
      [](OpPassManager &pm) {
        // Step 1-3: TOSA to Linalg (function-level passes).
        pm.nest<func::FuncOp>().addPass(
            mlir::tosa::createTosaInferShapesPass());
        pm.nest<func::FuncOp>().addPass(
            mlir::tosa::createTosaMakeBroadcastablePass());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToLinalgNamed());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToLinalg());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToArith());
        pm.nest<func::FuncOp>().addPass(mlir::tosa::createTosaToTensor());

        // Step 4: Linalg Generalization (function-level).
        pm.nest<func::FuncOp>().addPass(createLinalgGeneralizeNamedOpsPass());

        // Step 5: Canonicalization (module-level).
        pm.addPass(createCanonicalizerPass());

        // Step 6: Bufferization with proper options (module-level).
        OneShotBufferizationOptions bufferizationOptions;
        bufferizationOptions.bufferizeFunctionBoundaries = true;
        bufferizationOptions.setFunctionBoundaryTypeConversion(
            LayoutMapOption::IdentityLayoutMap);
        pm.addPass(createOneShotBufferizePass(bufferizationOptions));

        // Step 7: Linalg to Affine Loops (function-level).
        pm.nest<func::FuncOp>().addPass(createConvertLinalgToAffineLoopsPass());

        // Step 8: Cleanup subview and copy operations (function-level).
        pm.nest<func::FuncOp>().addPass(createFoldSubViewPass());
        pm.nest<func::FuncOp>().addPass(createConvertCopyToAffineLoopsPass());

        // Step 9: Final Affine cleanup (function-level).
        pm.nest<func::FuncOp>().addPass(
            mlir::affine::createAffineLoopNormalizePass());
        pm.nest<func::FuncOp>().addPass(
            mlir::affine::createSimplifyAffineStructuresPass());
        pm.addPass(createCanonicalizerPass());
      });
}