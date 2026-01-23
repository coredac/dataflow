add_mlir_library(MLIRTosaToTaskflowPipeline
  TosaToTaskflowPipeline.cpp

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRTaskflow
  MLIRAffineToTaskflowPass
  MLIRTosaToLinalg
  MLIRTosaToTensor
  MLIRTosaToArith
)
