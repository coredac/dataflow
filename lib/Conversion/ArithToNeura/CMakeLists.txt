set(LLVM_TARGET_DEFINITIONS
  ${CMAKE_CURRENT_SOURCE_DIR}/ArithToNeuraPatterns.td
)

mlir_tablegen(ArithToNeuraPatterns.inc
  -gen-rewriters
  -I ${MLIR_SOURCE_DIR}/include
  -I ${MLIR_BINARY_DIR}/include
  -I ${CMAKE_SOURCE_DIR}/include
  -I ${CMAKE_CURRENT_SOURCE_DIR}
)
add_public_tablegen_target(ArithToNeuraPatternGen)

add_mlir_library(NeuraArithToNeura
  ArithToNeuraPass.cpp

  DEPENDS
  ArithToNeuraPatternGen
  NeuraOpsIncGen
  NeuraDialectIncGen
  NeuraDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
)

target_include_directories(NeuraArithToNeura PUBLIC
    ${CMAKE_BINARY_DIR}/lib/NeuraDialect
    ${CMAKE_BINARY_DIR}/lib/Conversion/ArithToNeura
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(NeuraArithToNeura
  PRIVATE ${LLVM_DEFINITIONS}
)

