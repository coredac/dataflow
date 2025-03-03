# Adds MLIR dependencies.
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})
add_definitions(${MLIR_DEFINITIONS})

# Registers the dialect.
add_mlir_dialect(NeuraOps neura)
add_mlir_dialect(NeuraDialect neura)

# Lists all source files in this dialect.
set(SOURCES
  NeuraDialect.cpp
  NeuraOps.cpp
)

# Creates a single MLIR library target for the dialect.
add_mlir_library(NeuraDialect STATIC ${SOURCES}
  LINK_LIBS PUBLIC
    MLIRIR         # MLIR Core IR
    MLIRDialect    # MLIR Dialect
    MLIRSupport    # MLIR Support utilities
    MLIRTransforms # MLIR transformation passes
)

target_include_directories(NeuraDialect PUBLIC
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${MLIR_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}  # Add LLVM headers
  ${MLIR_INCLUDE_DIRS}  # Add MLIR headers
)

# Makes sure the compilation/build happens after tablegen.
add_dependencies(NeuraDialect NeuraOpsIncGen NeuraDialectIncGen)

