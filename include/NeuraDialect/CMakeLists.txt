# Set TableGen include paths
set(MLIR_TABLEGEN_INCLUDES 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/NeuraDialect
    ${CMAKE_CURRENT_BINARY_DIR}/include/NeuraDialect
    ${MLIR_MAIN_INCLUDE_DIR}
    ${MLIR_INCLUDE_DIR})

add_mlir_dialect(Neura neura)

set(LLVM_TARGET_DEFINITIONS NeuraPasses.td)
mlir_tablegen(NeuraPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRNeuraTransformsIncGen)