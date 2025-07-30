// NeuraOps.td - Custom operation definitions.

include "NeuraDialect/NeuraDialect.td"

// ----------------------------------------------------
// Defines basic scalar operations.

def Neura_ConstantOp : Op<NeuraDialect, "constant"> {
  let arguments = (ins 
    AnyAttr:$value,
    OptionalAttr<BoolAttr>:$predicate  // Add optional predicate attribute
  );
  let results = (outs AnyType:$result);
  // let assemblyFormat = "attr-dict `:` type($result)";
}

// Defines an addition operation.
def Neura_AddOp : Op<NeuraDialect, "add"> {
  let summary = "Integer addition operation";
  let opName = "add";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_SubOp : Op<NeuraDialect, "sub"> {
  let summary = "Integer substraction operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_MulOp : Op<NeuraDialect, "mul"> {
  let summary = "Integer multiplication operation";
  let opName = "mul";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_DivOp : Op<NeuraDialect, "div"> {
  let summary = "Integer division operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point addition operation.
def Neura_FAddOp : Op<NeuraDialect, "fadd"> {
  let summary = "Floating addition operation";
  let opName = "fadd";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point substraction operation.
def Neura_FSubOp: Op<NeuraDialect, "fsub"> {
  let summary = "Floating substraction operation";
  let opName = "fsub";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point multiplication operation.
def Neura_FMulOp : Op<NeuraDialect, "fmul"> {
  let summary = "Floating multiplication operation";
  let opName = "fmul";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

def Neura_FDivOp : Op<NeuraDialect, "fdiv"> {
  let summary = "Floating division operation";
  let arguments = (ins AnyFloat:$lhs, AnyFloat:$rhs, Optional<AnyType>:$predicate);
  let results = (outs AnyFloat:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
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
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Optional<AnyType>:$predicate,
                   StrAttr:$cmpType);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $cmpTypeAttr `,` $cmp_type attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point compare operation.
def Neura_FCmpOp : Op<NeuraDialect, "fcmp"> {
  let summary = "Floating-point compare operation";
  let opName = "fcmp";
  let arguments = (ins AnyType:$lhs,
                   AnyType:$rhs,
                   Optional<AnyType>:$predicate,
                   StrAttr:$cmpType);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $cmpType attr-dict `:` type($result)";
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

// Defines a load operation with integrated address calculation.
def Neura_LoadIndexedOp: Op<NeuraDialect, "load_indexed", [AttrSizedOperandSegments]>{
  let summary = "Load with integrated address calculation for multi-dimensional arrays";
  let description = [{
    Calculates the address using the base address and indices.
    Load the value at the calculated address.
    Example:
      %value = neura.load_indexed %base [%arg1, %arg2] : f32
  }];
  let arguments = (ins AnyType:$base, Variadic<AnyType>:$indices, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `[` $indices `:` type($indices) `]` type($base) ($predicate^ `:` type($predicate))? attr-dict `:` type($result)";
}

//Defines a store operation with integrated address calculation.
def Neura_StoreIndexedOp: Op<NeuraDialect, "store_indexed", [AttrSizedOperandSegments]> {
  let summary = "Store with integrated address calculation for multi-dimensional arrays";
  let description = [{
    Calculates the address using the base address and indices.
    Store the value at the calculated address.
    Example:
      neura.store_indexed %value, %base [%arg1, %arg2] : f32
  }];
  let arguments = (ins AnyType:$value, AnyType:$base, Variadic<AnyType>:$indices, Optional<AnyType>:$predicate);
  let results = (outs);
  let assemblyFormat = "$value `to` $base `[` $indices `:` type($indices) `]` type($base) ($predicate^ `:` type($predicate))? attr-dict `:` type($value)";
}

// Defines a pointer computation operation.
def Neura_GEP : Op<NeuraDialect, "gep"> {
  let summary = "Pointer computation using offset indices";
  let arguments = (ins AnyType:$base, Variadic<AnyType>:$indicesAndPredicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$base `[` $indicesAndPredicate `]` `,` $predicate attr-dict";
}

// Defines a conditional branch operation.
def Neura_CondBr : Op<NeuraDialect, "cond_br", [Terminator, AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$condition,
                   Optional<AnyType>:$predicate,
                   Variadic<AnyType>:$trueArgs,
                   Variadic<AnyType>:$falseArgs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$condition `:` type($condition) ($predicate^ `:` type($predicate))? `then` ($trueArgs^ `:` type($trueArgs))? `to` $trueDest `else` ($falseArgs^ `:` type($falseArgs))? `to` $falseDest attr-dict";
}

// Defines an unconditional branch operation.
def Neura_Br : Op<NeuraDialect, "br", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "($args^ `:` type($args))? `to` $dest attr-dict";
}

def Neura_SelOp : Op<NeuraDialect, "sel"> {
  let arguments = (ins AnyType:$ifTrue, AnyType:$ifFalse, AnyType:$cond);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$ifTrue `,` $ifFalse `,` $cond attr-dict `:` type($ifTrue)";
}

def Neura_NotOp : Op<NeuraDialect, "not"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  // let assemblyFormat = "$input attr-dict `:` type($output)";
}

// Defines a return operation.
def Neura_ReturnOp : Op<NeuraDialect, "return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  // let assemblyFormat = "($values^)? `,` $predicate attr-dict";
}

// Defines a cast operation for type conversion.
def Neura_CastOp : Op<NeuraDialect, "cast">{
  let summary = "Generic type conversion operation";
  let arguments = (ins AnyType:$input, StrAttr:$cast_type, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$input type($input) `->` type($output) `,` $predicate attr-dict";
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
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_FMulFAddOp : Op<NeuraDialect, "fmul_fadd"> {
  let summary = "Fused fadd(fmul(a, b), c)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c, Optional<AnyType>:$predicate);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// ----------------------------------------------------
// Defines move operations.
def Neura_DataMovOp : Op<NeuraDialect, "data_mov"> {
  let summary = "Data movement operation";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  // let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// ----------------------------------------------------
// Defines ctrl-related operations.

// Phi operation for merging values in dataflow form
def Neura_PhiOp : Op<NeuraDialect, "phi"> {
  let summary = "Phi node in dataflow form";
  let description = [{
    Merges values from different control paths in dataflow form.
    Used with reserve and ctrl_mov to represent control flow.
    
    Example:
      %v = neura.reserve : f32           // Create placeholder
      %result = neura.phi %init, %v      // Merge initial and loop-carried values
      neura.ctrl_mov %next to %v         // Connect next iteration
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);

  // Explicitly specify types for operands in the assembly format
  // let assemblyFormat = "$init_val `:` type($init_val) `,` $loop_val `:` type($loop_val) attr-dict `,` type($result)";
}

// Control movement extending base move but with different signature.
def Neura_CtrlMovOp : Op<NeuraDialect, "ctrl_mov"> {
  let summary = "Control movement operation";
  let description = [{
    Connects a value to a reserved placeholder in the dataflow.
    Used to establish control flow dependencies.

    Example:
      ctrl_mov %value to %placeholder : f32  // Connect value to placeholder
  }];

  // Add type constraints for both operands
  let arguments = (ins AnyType:$value, AnyType:$target);
  let results = (outs);

  // Correct assembly format - types must be space-separated
  let assemblyFormat = "$value `->` $target attr-dict `:` type($value) type($target)";
}

// Reserve operation for control flow values.
def Neura_ReserveOp : Op<NeuraDialect, "reserve"> {
  let summary = "Creates a placeholder for control flow values";
  let description = [{
    Creates a placeholder value that will be connected via ctrl_mov.
    Used to represent control flow dependencies in dataflow form.
    
    Example:
      %v = neura.reserve : f32      // Create placeholder
      %result = neura.phi %init, %v // Use in phi node
      neura.ctrl_mov %next to %v    // Connect value
  }];

  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Neura_GrantPredicateOp : Op<NeuraDialect, "grant_predicate"> {
  let summary = "Grants a new predicate to a predicated value.";
  let description = [{
    Takes a predicated value and a predicate (i1), producing a new predicated
    value whose predicate bit is set to the given condition.
    
    Example:
      %g = neura.grant_predicate %val, %pred : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
  }];

  let arguments = (ins AnyType:$value, AnyType:$predicate);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value `,` $predicate attr-dict `:` type($value) `,` type($predicate) `->` type($result)";
}

def Neura_GrantOnceOp : Op<NeuraDialect, "grant_once"> {
  let summary = "Marks a value as valid once.";
  let description = [{
    Either grants a value a one-time predicate, or creates a constant value with a one-time predicate. The resulting value is considered valid
    only during its first activation. Used to initialize recurrence cycles.
    
    Example:
      %v = neura.grant_once %init : !neura.data<f32, i1> -> !neura.data<f32, i1>
      %c = neura.grant_once <{value = 42 : i64}> : !neura.data<i64, i1>
  }];

  let arguments = (ins 
  Optional<AnyType>:$value,
  OptionalAttr<AnyAttr>:$constant_value);
  let results = (outs AnyType:$result);

  // let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def Neura_GrantAlwaysOp : Op<NeuraDialect, "grant_always"> {
  let summary = "Marks a value as valid always.";
  let description = [{
    Grants a value always-valid predicate: the resulting value is considered valid
    during the entire application lifetime.
    
    Example:
      %v = neura.grant_always %init : !neura.data<f32, i1> -> !neura.data<f32, i1>
  }];

  let arguments = (ins 
  Optional<AnyType>:$value,
  OptionalAttr<AnyAttr>:$constant_value);
  let results = (outs AnyType:$result);

  // let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

// ----------------------------------------------------
// Defines fused control flow operations.

def Neura_LoopControlOp : Op<NeuraDialect, "loop_control">{
  let summary = "Generates loop indicies and valid predicates.";
  let description = [{
    Controls loop execution with minimal recurrence cycle length (1).
    Takes the current index and produces the next index and validity predicate.
    
    This operation combines comparison and increment logic into a single operation,
    enabling efficient loop execution on dataflow architectures with modulo scheduling.
    
    The first iteration automatically uses the start value because the reserve operation's
    predicate bit is initially 0, while the start value's predicate bit is 1.
    
    Example:
      // Loop control that calculates next index and validity in one step
      %next_idx, %loop_valid = neura.loop_control(
        parent_valid = %parent_valid,
        start = %start_val,
        end = %end_val,
        step = %step_val
      ) <{iteration_type = "increment"}> : !neura.data<i1, i1>,
                                           !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>
                                           -> !neura.data<i64, i1>, !neura.data<i1, i1>
  }];

  let arguments = (ins
    AnyType:$parentValid,  // Valid predicate from the parent loop.
    StrAttr:$iterationType, // Type of the loop iteration (e.g., "increment", "decrement").
    AnyType:$start,         // Start index of the loop (optional if startValue attr is presented).
    AnyType:$end,           // End index of the loop (optional if endValue attr is presented).
    AnyType:$step           // Step size for the loop (optional if stepValue attr is presented).
  );

  let results = (outs
    AnyType:$nextindex,         // Current loop index
    AnyType:$valid          // Valid predicate for the current index
  );

  let assemblyFormat =
    " `(``parent_valid` `=` $parentValid `,` `start` `=` $start `,` `end` `=` $end `,` `step` `=` $step`)` attr-dict `:` type($parentValid) `,` type($start) `,` type($end) `,` type($step) `->` type($nextindex) `,` type($valid)";
}