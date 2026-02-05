include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTaskflowOptimization
  AffineLoopTreeSerializationPass.cpp
  AffineLoopPerfectionPass.cpp

  DEPENDS
  MLIRTaskflowTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTaskflow
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
)