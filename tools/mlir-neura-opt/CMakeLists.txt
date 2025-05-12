# tools/mlir-neura-opt/CMakeLists.txt

add_executable(mlir-neura-opt
  mlir-neura-opt.cpp
)

# Links MLIR libraries.
target_link_libraries(mlir-neura-opt PRIVATE
  MLIROptLib        # MLIR optimizer library
  MLIRIR            # MLIR Core IR
  MLIRSupport       # MLIR Support utilities
  MLIRTransforms    # MLIR transformation passes
  MLIRDialect       # MLIR Dialect
  NeuraDialect      # Custom dialect
  MLIRFuncDialect   # Builtin dialect required by custom dialect
  MLIRArithDialect
  NeuraArithToNeura
)

# Includes directories.
target_include_directories(mlir-neura-opt PRIVATE
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${MLIR_SOURCE_DIR}/include
  # Adds LLVM headers.
  ${LLVM_INCLUDE_DIRS}
  # Adds MLIR headers.
  ${MLIR_INCLUDE_DIRS}
)

# Adds dependencies.
add_dependencies(mlir-neura-opt NeuraOpsIncGen NeuraDialectIncGen)

