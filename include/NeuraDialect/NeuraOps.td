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
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyInteger:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point addition operation.
def Neura_FAddOp : Op<NeuraDialect, "fadd"> {
  let summary = "Floating addition operation";
  let opName = "fadd";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point multiplication operation.
def Neura_FMulOp : Op<NeuraDialect, "fmul"> {
  let summary = "Floating multiplication operation";
  let opName = "fmul";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a bitwise OR operation.
def Neura_OrOp : Op<NeuraDialect, "or"> {
  let summary = "Bitwise OR operation";
  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnySignlessInteger:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines an integer compare operation.
def Neura_ICmpOp : Op<NeuraDialect, "icmp"> {
  let summary = "Integer compare operation";
  let opName = "icmp";
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs, Optional<AnyType>:$predicate,
                   StrAttr:$cmpType);
  let results = (outs I1:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $cmpTypeAttr `,` $cmp_type attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

// Defines a load operation.
def Neura_LoadOp : Op<NeuraDialect, "load"> {
  let arguments = (ins AnyType:$addr, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$value);
  // let assemblyFormat = "$addr `,` $predicate attr-dict `:` type($value)";
}

// Defines a store operation.
def Neura_StoreOp : Op<NeuraDialect, "store"> {
  let arguments = (ins AnyType:$value, AnyType:$addr, Optional<AnyType>:$predicate);
  let results = (outs);
  // let assemblyFormat = "$value `,` $addr `,` $predicate attr-dict";
}

// Defines a pointer computation operation.
def Neura_GEP : Op<NeuraDialect, "gep"> {
  let summary = "Pointer computation using offset indices";
  let arguments = (ins AnyType:$base, Variadic<AnyInteger>:$indicesAndPredicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$base `[` $indicesAndPredicate `]` `,` $predicate attr-dict";
}

// Defines a conditional branch operation.
def Neura_CondBr : Op<NeuraDialect, "cond_br", [Terminator, AttrSizedOperandSegments]> {
  let arguments = (ins I1:$condition,
                   Optional<AnyType>:$predicate,
                   Variadic<AnyType>:$trueArgs,
                   Variadic<AnyType>:$falseArgs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$condition ($predicate^ `:` type($predicate))? `then` $trueArgs `:` type($trueArgs) `to` $trueDest `else` $falseArgs `:` type($falseArgs) `to` $falseDest attr-dict";
}

// Defines a return operation.
def Neura_ReturnOp : Op<NeuraDialect, "return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  // let assemblyFormat = "($values^)? `,` $predicate attr-dict";
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
  let arguments = (ins VectorOfAnyFloat:$lhs, VectorOfAnyFloat:$rhs, Optional<AnyType>:$predicate);
  let results = (outs VectorOfAnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// ----------------------------------------------------
// Defines fused operations.

def Neura_FAddFAddOp : Op<NeuraDialect, "fadd_fadd"> {
  let summary = "Fused fadd(fadd(a, b), c)";
  let arguments = (ins AnyFloat:$a, AnyFloat:$b, AnyFloat:$c, Optional<AnyType>:$predicate);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_FMulFAddOp : Op<NeuraDialect, "fmul_fadd"> {
  let summary = "Fused fadd(fmul(a, b), c)";
  let arguments = (ins AnyFloat:$a, AnyFloat:$b, AnyFloat:$c, Optional<AnyType>:$predicate);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a move operation for data communication.
def Neura_MovOp : Op<NeuraDialect, "mov"> {
  let summary = "Move operation";
  let opName = "mov";
  let arguments = (ins AnyType:$lhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $predicate attr-dict `:` type($lhs) `->` type($result)";
  // let traits = [Pure];
}