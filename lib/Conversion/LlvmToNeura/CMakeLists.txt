add_mlir_library(NeuraLlvmToNeura
  LlvmToNeuraPass.cpp

  DEPENDS
  NeuraOpsIncGen
  NeuraDialectIncGen
  NeuraDialect

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
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(NeuraLlvmToNeura
  PRIVATE ${LLVM_DEFINITIONS}
)
