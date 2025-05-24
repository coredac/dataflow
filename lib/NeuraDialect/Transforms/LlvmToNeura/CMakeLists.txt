set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/LlvmToNeuraPatterns.td
)
mlir_tablegen(LlvmToNeuraPatterns.inc
  -gen-rewriters
  -I ${MLIR_SOURCE_DIR}/include
  -I ${MLIR_BINARY_DIR}/include
  -I ${CMAKE_SOURCE_DIR}/include
  -I ${CMAKE_CURRENT_SOURCE_DIR}
)
add_public_tablegen_target(MLIRLlvmToNeuraPatternGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(MLIRNeuraLlvmToNeuraPass
  LlvmToNeuraPass.cpp

  DEPENDS
  MLIRLlvmToNeuraPatternGen
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
