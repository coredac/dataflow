set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/ArithToNeuraPatterns.td)
mlir_tablegen(ArithToNeuraPatterns.inc
  -gen-rewriters
  -I ${MLIR_SOURCE_DIR}/include
  -I ${MLIR_BINARY_DIR}/include
  -I ${CMAKE_SOURCE_DIR}/include
  -I ${CMAKE_CURRENT_SOURCE_DIR}
)
add_public_tablegen_target(MLIRNeuraArithToNeuraIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(MLIRNeuraArithToNeuraPass
  ArithToNeuraPass.cpp

  DEPENDS
  MLIRNeuraArithToNeuraIncGen
  MLIRNeuraTransformsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    # MLIRNeura
)