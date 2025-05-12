include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(MLIR_TBLGEN_INCLUDES
  ${MLIR_SOURCE_DIR}/include  # MLIR source include directory
  ${MLIR_BINARY_DIR}/include  # MLIR build include directory
)

