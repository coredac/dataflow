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

// Defines a move operation for data communication.
def Neura_MovOp : Op<NeuraDialect, "mov"> {
  let summary = "Move operation";
  let opName = "mov";
  let arguments = (ins AnyType:$lhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs attr-dict `:` type($lhs) `->` type($result)";
  // let traits = [Pure];
}
