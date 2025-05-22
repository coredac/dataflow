include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "NeuraDialect/NeuraOps.td"

def : Pat<
  (LLVM_FAddOp $lhs, $rhs, $_fastmath),
  (Neura_FAddOp $lhs, $rhs)
>;

def : Pat<
  (LLVM_ConstantOp $value),
  (Neura_ConstantOp $value)
>;

