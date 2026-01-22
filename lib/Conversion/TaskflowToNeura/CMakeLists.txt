include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTaskflowToNeuraPass
  TaskflowToNeuraPass.cpp

  DEPENDS
  MLIRConversionIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTaskflow
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRNeura
  MLIRSupport
)
