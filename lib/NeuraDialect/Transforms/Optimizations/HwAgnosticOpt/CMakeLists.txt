include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRNeuraHwAgnosticOpt
  FoldConstantPass.cpp

  DEPENDS
  MLIRNeuraTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRNeura
  MLIRSupport
)