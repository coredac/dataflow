include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "NeuraDialect/NeuraOps.td"

// Floating point binary operations
def : Pat<
  (LLVM_FAddOp $lhs, $rhs, $_fastmath),
  (Neura_FAddOp $lhs, $rhs)
>;

def : Pat<
  (LLVM_FSubOp $lhs, $rhs, $_fastmath),
  (Neura_FSubOp $lhs, $rhs)
>;

def : Pat<
  (LLVM_ConstantOp $value),
  (Neura_ConstantOp $value)
>;

def : Pat<
  (LLVM_OrOp $lhs, $rhs),
  (Neura_OrOp $lhs, $rhs)
>;

