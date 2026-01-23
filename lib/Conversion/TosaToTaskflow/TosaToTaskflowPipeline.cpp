#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "Conversion/ConversionPasses.h"

using namespace mlir;

namespace {
void buildTosaToTaskflowPipeline(OpPassManager &pm) {
  // 1. TOSA to Linalg/Arith/Tensor
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToArith());
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());

  // 2. Linalg optimizations
  pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<func::FuncOp>(createConvertTensorToLinalgPass());

  // 3. One-shot bufferization
  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  bufOpts.functionArgTypeConverterFn = [](TensorType tensorType, Attribute memorySpace,
                                          func::FuncOp funcOp, const bufferization::BufferizationOptions &options) {
    return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace);
  };
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(createCanonicalizerPass());

  // 4. Linalg to Affine
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
  pm.addNestedPass<func::FuncOp>(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // 5. Affine to Taskflow
  pm.addPass(createConvertAffineToTaskflowPass());
}
} // namespace

void mlir::registerTosaToTaskflowPipeline() {
  PassPipelineRegistration<>(
      "tosa-to-taskflow-pipeline",
      "Lower TOSA to Taskflow dialect through Linalg and Affine.",
      buildTosaToTaskflowPipeline);
}
