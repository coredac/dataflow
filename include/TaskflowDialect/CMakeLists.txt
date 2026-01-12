add_mlir_dialect(Taskflow taskflow)

set(LLVM_TARGET_DEFINITIONS TaskflowPasses.td)
mlir_tablegen(TaskflowPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTaskflowTransformsIncGen)