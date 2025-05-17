# Adds MLIR dependencies.
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})
add_definitions(${MLIR_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../include/NeuraDialect/NeuraOps.td)

mlir_tablegen(NeuraOps.h.inc -gen-op-decls
  -I${MLIR_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(NeuraOps.cpp.inc -gen-op-defs
  -I${MLIR_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(NeuraOpsIncGen)

mlir_tablegen(NeuraDialect.h.inc -gen-dialect-decls
  -dialect=neura
  -I${MLIR_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/include)
mlir_tablegen(NeuraDialect.cpp.inc -gen-dialect-defs
  -dialect=neura
  -I${MLIR_SOURCE_DIR}/include
  -I${CMAKE_SOURCE_DIR}/include)
add_public_tablegen_target(NeuraDialectIncGen)

add_public_tablegen_target(MLIRNeuraIncGen)

# Creates a single MLIR library target for the dialect.
add_mlir_library(NeuraDialect STATIC # ${SOURCES}
  NeuraDialect.cpp
  NeuraOps.cpp

  DEPENDS
    NeuraOpsIncGen
    NeuraDialectIncGen

  LINK_LIBS PUBLIC
    MLIRIR         # MLIR Core IR
    MLIRDialect    # MLIR Dialect
    MLIRSupport    # MLIR Support utilities
    MLIRTransforms # MLIR transformation passes
)

include_directories(${CMAKE_SOURCE_DIR}/include)

target_include_directories(NeuraDialect PUBLIC
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_BINARY_DIR}/lib/NeuraDialect
  ${CMAKE_SOURCE_DIR}/include
  ${MLIR_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}  # Add LLVM headers
  ${MLIR_INCLUDE_DIRS}  # Add MLIR headers
)

# Makes sure the compilation/build happens after tablegen.
add_dependencies(NeuraDialect NeuraOpsIncGen NeuraDialectIncGen)

