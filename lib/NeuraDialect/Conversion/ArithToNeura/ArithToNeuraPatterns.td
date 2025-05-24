include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "NeuraDialect/NeuraOps.td"

def : Pat<
  (Arith_AddFOp $lhs, $rhs, $_fastmath),
  (Neura_FAddOp $lhs, $rhs)
>;

