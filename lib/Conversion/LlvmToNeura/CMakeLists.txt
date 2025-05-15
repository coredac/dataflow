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
add_public_tablegen_target(LlvmToNeuraPatternGen)

add_mlir_library(NeuraLlvmToNeura
  LlvmToNeuraPass.cpp

  DEPENDS
  NeuraOpsIncGen
  NeuraDialectIncGen
  NeuraDialect
  LlvmToNeuraPatternGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

target_include_directories(NeuraLlvmToNeura PUBLIC
    ${CMAKE_BINARY_DIR}/lib/NeuraDialect
    ${CMAKE_BINARY_DIR}/lib/Conversion/LlvmToNeura
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(NeuraLlvmToNeura
  PRIVATE ${LLVM_DEFINITIONS}
)
