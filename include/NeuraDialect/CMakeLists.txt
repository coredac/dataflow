include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(MLIR_TBLGEN_INCLUDES
  ${MLIR_SOURCE_DIR}/include  # MLIR source include directory
  ${MLIR_BINARY_DIR}/include  # MLIR build include directory
)

set(LLVM_TARGET_DEFINITIONS NeuraOps.td)
mlir_tablegen(NeuraOps.h.inc -gen-op-decls -dialect=neura -I${MLIR_SOURCE_DIR}/include)
mlir_tablegen(NeuraOps.cpp.inc -gen-op-defs -dialect=neura -I${MLIR_SOURCE_DIR}/include)
add_public_tablegen_target(NeuraOpsIncGen)

set(LLVM_TARGET_DEFINITIONS NeuraDialect.td)
mlir_tablegen(NeuraDialect.h.inc -gen-dialect-decls -dialect=neura -I${MLIR_SOURCE_DIR}/include)
add_public_tablegen_target(NeuraDialectIncGen)

