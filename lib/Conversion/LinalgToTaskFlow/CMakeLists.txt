include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRLinalgToTaskFlowPass
  LinalgToTaskFlowPass.cpp

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
  MLIRTaskFlow
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
)