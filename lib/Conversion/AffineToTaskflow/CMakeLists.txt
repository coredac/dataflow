include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRAffineToTaskflowPass
  AffineToTaskflowPass.cpp
  FoldSubViewPass.cpp
  ConvertCopyToAffineLoopsPass.cpp

  DEPENDS
  MLIRConversionIncGen

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