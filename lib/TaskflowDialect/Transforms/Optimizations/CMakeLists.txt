include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTaskflowOptimization
  AffineLoopTreeSerializationPass.cpp
  AffineLoopPerfectionPass.cpp
  MemoryAccessStreamingFusion.cpp
  ResourceAwareTaskOptimizationPass.cpp

  DEPENDS
  MLIRTaskflowTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTaskflow
  MLIRTaskflowUtil
  MLIRTaskflowTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRNeura
  MLIRNeuraTransforms
  MLIRConversion
  MLIRSupport
)