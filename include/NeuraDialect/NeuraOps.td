// NeuraOps.td - Custom operation definitions.
include "mlir/IR/OpBase.td"
include "NeuraDialect/NeuraDialect.td"

// Defines an addition operation.
def Neura_AddOp : Op<NeuraDialect, "add"> {
  let summary = "Addition operation";
  let opName = "add";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  // let traits = [Pure];
}
