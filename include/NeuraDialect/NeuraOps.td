// NeuraOps.td - Custom operation definitions.

include "NeuraDialect/NeuraDialect.td"

// ----------------------------------------------------
// Defines basic scalar operations.

def Neura_ConstantOp : Op<NeuraDialect, "constant"> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "attr-dict `:` type($result)";
}

// Defines an addition operation.
def Neura_AddOp : Op<NeuraDialect, "add"> {
  let summary = "Integer addition operation";
  let opName = "add";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point addition operation.
def Neura_FAddOp : Op<NeuraDialect, "fadd"> {
  let summary = "Floating addition operation";
  let opName = "fadd";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point multiplication operation.
def Neura_FMulOp : Op<NeuraDialect, "fmul"> {
  let summary = "Floating multiplication operation";
  let opName = "fmul";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a move operation for data communication.
def Neura_MovOp : Op<NeuraDialect, "mov"> {
  let summary = "Move operation";
  let opName = "mov";
  let arguments = (ins AnyType:$lhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs attr-dict `:` type($lhs) `->` type($result)";
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

def Neura_GEP : Op<NeuraDialect, "gep"> {
  let summary = "Pointer computation using offset indices";
  let arguments = (ins AnyType:$base, Variadic<AnyInteger>:$indices);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$base `[` $indices `]` attr-dict";
}

def Neura_CondBr : Op<NeuraDialect, "cond_br", [Terminator, AttrSizedOperandSegments]> {
  let arguments = (ins I1:$condition,
                   Variadic<AnyType>:$trueArgs,
                   Variadic<AnyType>:$falseArgs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$condition `then` $trueArgs `:` type($trueArgs) `to` $trueDest `else` $falseArgs `:` type($falseArgs) `to` $falseDest attr-dict";
}

def Neura_ReturnOp : Op<NeuraDialect, "return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  // let assemblyFormat = "($values^)? attr-dict";
}

// ----------------------------------------------------
// Defines vector operations.

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

// ----------------------------------------------------
// Defines fused operations.

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


