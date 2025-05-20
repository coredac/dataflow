// NeuraOps.td - Custom operation definitions.

include "NeuraDialect/NeuraDialect.td"

// Defines an addition operation.
def Neura_AddOp : Op<NeuraDialect, "add"> {
  let summary = "Integer addition operation";
  let opName = "add";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines an addition operation.
def Neura_FAddOp : Op<NeuraDialect, "fadd"> {
  let summary = "Floating addition operation";
  let opName = "fadd";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a multiplication operation.
def Neura_FMulOp : Op<NeuraDialect, "fmul"> {
  let summary = "Floating multiplication operation";
  let opName = "fmul";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def VectorOfAnyFloat :
  TypeConstraint<
    CPred<
      "mlir::isa<::mlir::VectorType>($_self) && "
      "mlir::isa<::mlir::FloatType>(mlir::cast<::mlir::VectorType>($_self).getElementType())"
    >,
    "vector of floats"
  >;

// Defines a vector multiplication operation.
def Neura_VFMulOp : Op<NeuraDialect, "vfmul"> {
  let summary = "Vector floating multiplication operation";
  let opName = "vfmul";
  let arguments = (ins VectorOfAnyFloat:$lhs, VectorOfAnyFloat:$rhs);
  let results = (outs VectorOfAnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_FAddFAddOp : Op<NeuraDialect, "fadd_fadd"> {
  let summary = "Fused fadd(fadd(a, b), c)";
  let arguments = (ins AnyFloat:$a, AnyFloat:$b, AnyFloat:$c);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_FMulFAddOp : Op<NeuraDialect, "fmul_fadd"> {
  let summary = "Fused fadd(fmul(a, b), c)";
  let arguments = (ins AnyFloat:$a, AnyFloat:$b, AnyFloat:$c);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
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

def Neura_ICmpOp : Op<NeuraDialect, "icmp"> {
  let summary = "Integer compare operation";
  let opName = "icmp";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs,
                   StrAttr:$predicate);
  let results = (outs I1:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

def Neura_LoadOp : Op<NeuraDialect, "load"> {
  let arguments = (ins AnyType:$addr);
  let results = (outs AnyType:$value);
  // let assemblyFormat = "$addr attr-dict `:` type($value)";
}

def Neura_StoreOp : Op<NeuraDialect, "store"> {
  let arguments = (ins AnyType:$value, AnyType:$addr);
  let results = (outs);
  // let assemblyFormat = "$value `,` $addr attr-dict";
}

