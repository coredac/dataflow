include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "NeuraDialect/NeuraOps.td"

// Floating point binary operations.
// Deprecated Pattern: Because we need the predicate bit to be set to null initially
// def : Pat<
//   (LLVM_FSubOp $lhs, $rhs, $_fastmath),
//   (Neura_FSubOp $lhs, $rhs)
// >;

