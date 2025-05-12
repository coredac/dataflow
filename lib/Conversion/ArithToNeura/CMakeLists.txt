add_mlir_library(NeuraArithToNeura
  ArithToNeura.cpp

  DEPENDS
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
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(NeuraArithToNeura
  PRIVATE ${LLVM_DEFINITIONS}
)
