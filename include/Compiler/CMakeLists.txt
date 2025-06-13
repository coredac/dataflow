set(LLVM_TARGET_DEFINITIONS CompilerPasses.td)
mlir_tablegen(CompilerPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRCompilerPassesIncGen)