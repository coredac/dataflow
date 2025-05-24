add_mlir_dialect(Neura neura)

set(LLVM_TARGET_DEFINITIONS NeuraPasses.td)
mlir_tablegen(NeuraPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRNeuraTransformsIncGen)