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
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_SubOp : Op<NeuraDialect, "sub"> {
  let summary = "Integer substraction operation";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_MulOp : Op<NeuraDialect, "mul"> {
  let summary = "Integer multiplication operation";
  let opName = "mul";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_DivOp : Op<NeuraDialect, "div"> {
  let summary = "Integer division operation";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_RemOp : Op<NeuraDialect, "rem">{
  let summary = "Integer remainder operation";
  let description = [{
    Performs an integer remainder operation, computing the result of
    a % b, where % is the remainder operator.

    Example:
      %result = neura.rem %a, %b : i32
  }];
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point addition operation.
def Neura_FAddOp : Op<NeuraDialect, "fadd"> {
  let summary = "Floating addition operation";
  let opName = "fadd";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $predicate attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point substraction operation.
def Neura_FSubOp: Op<NeuraDialect, "fsub"> {
  let summary = "Floating substraction operation";
  let opName = "fsub";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines a floating-point multiplication operation.
def Neura_FMulOp : Op<NeuraDialect, "fmul"> {
  let summary = "Floating multiplication operation";
  let opName = "fmul";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

def Neura_FDivOp : Op<NeuraDialect, "fdiv"> {
  let summary = "Floating division operation";
  let description = [{
    Performs a floating-point division operation, computing the result of
    a / b, where / is the floating-point division operator.

    Example:
      %result = neura.fdiv %a, %b : f32
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let traits = [SameOperandsAndResultElementType];
}

// Defines a bitwise OR operation.
def Neura_OrOp : Op<NeuraDialect, "or"> {
  let summary = "Bitwise OR operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// Defines an integer compare operation.
def Neura_ICmpOp : Op<NeuraDialect, "icmp"> {
  let summary = "Integer compare operation";
  let opName = "icmp";
  let arguments = (ins AnyType:$lhs, Optional<AnyType>:$rhs,
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
                   Optional<AnyType>:$rhs,
                   StrAttr:$cmpType);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$lhs `,` $rhs `,` $cmpType attr-dict `:` type($result)";
  // let traits = [SameOperandsAndResultElementType];
}

// Defines a load operation.
def Neura_LoadOp : Op<NeuraDialect, "load"> {
  let arguments = (ins AnyType:$addr);
  let results = (outs AnyType:$value);
  // let assemblyFormat = "$addr `,` attr-dict `:` type($value)";
}

// Defines a store operation.
def Neura_StoreOp : Op<NeuraDialect, "store"> {
  let arguments = (ins AnyType:$value, AnyType:$addr);
  let results = (outs);
  // let assemblyFormat = "$value `,` $addr `,` attr-dict";
}

// Defines a load operation with integrated address calculation.
def Neura_LoadIndexedOp: Op<NeuraDialect, "load_indexed">{
  let summary = "Load with integrated address calculation for multi-dimensional arrays";
  let description = [{
    Calculates the address using the base address and indices.
    Load the value at the calculated address.
    Example:
      %value = neura.load_indexed %base [%arg1, %arg2] : f32
  }];
  let arguments = (ins AnyType:$base, Variadic<AnyType>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$base `[` $indices `:` type($indices) `]` type($base) attr-dict `:` type($result)";
}

//Defines a store operation with integrated address calculation.
def Neura_StoreIndexedOp: Op<NeuraDialect, "store_indexed"> {
  let summary = "Store with integrated address calculation for multi-dimensional arrays";
  let description = [{
    Calculates the address using the base address and indices.
    Store the value at the calculated address.
    Example:
      neura.store_indexed %value, %base [%arg1, %arg2] : f32
  }];
  let arguments = (ins AnyType:$value, AnyType:$base, Variadic<AnyType>:$indices);
  let results = (outs);
  let assemblyFormat = "$value `to` $base `[` $indices `:` type($indices) `]` type($base) attr-dict `:` type($value)";
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
                   Variadic<AnyType>:$trueArgs,
                   Variadic<AnyType>:$falseArgs);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = "$condition `:` type($condition) `then` ($trueArgs^ `:` type($trueArgs))? `to` $trueDest `else` ($falseArgs^ `:` type($falseArgs))? `to` $falseDest attr-dict";
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
  // let assemblyFormat = "($values^)? `,` attr-dict";
}

// Defines a cast operation for type conversion.
def Neura_CastOp : Op<NeuraDialect, "cast">{
  let summary = "Generic type conversion operation";
  let arguments = (ins AnyType:$input, StrAttr:$cast_type);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$input type($input) `->` type($output) `,` attr-dict";
}

// Defines an alloca operation for memory allocation.
def Neura_AllocaOp : Op<NeuraDialect, "alloca"> {
  let summary = "Memory allocation operation";
  let description = [{
    Allocates memory on the stack, similar to llvm.alloca.
    Takes a predicated size value and returns a pointer to the allocated memory.

    Example:
      %ptr = neura.alloca %size : !neura.data<i32, i1> -> !llvm.ptr
  }];

  let arguments = (ins Optional<AnyType>:$size);
  let results = (outs AnyType:$result);
  let assemblyFormat = "($size^ `:` type($size))? attr-dict `->` type($result)";
}

// Defines a sign extension operation.
def Neura_SExtOp : Op<NeuraDialect, "sext"> {
  let summary = "Sign extension operation";
  let description = [{
    Sign extends a value from a smaller integer type to a larger integer type.
    Similar to llvm.sext, but works with predicated values.

    Example:
      %extended = neura.sext %value : !neura.data<i8, i1> -> !neura.data<i32, i1>
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

// Defines a zero extension operation.
def Neura_ZExtOp : Op<NeuraDialect, "zext"> {
  let summary = "Zero extension operation";
  let description = [{
    Zero extends a value from a smaller integer type to a larger integer type.
    Similar to llvm.zext, but works with predicated values.

    Example:
      %extended = neura.zext %value : !neura.data<i8, i1> -> !neura.data<i32, i1>
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

// Defines a logical shift left operation.
def Neura_ShlOp : Op<NeuraDialect, "shl"> {
  let summary = "Logical shift left operation";
  let description = [{
    Performs a logical left shift on an integer value.
    Similar to llvm.shl, but works with predicated values.

    Example:
      %shifted = neura.shl %value, %shiftAmount : !neura.data<i32, i1> -> !neura.data<i32, i1>
  }];
  let arguments = (ins AnyType:$value, Optional<AnyType>:$shiftAmount);
  let results = (outs AnyType:$result);
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
  // let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

// ----------------------------------------------------
// Defines fused operations.

def Neura_FAddFAddOp : Op<NeuraDialect, "fadd_fadd"> {
  let summary = "Fused fadd(fadd(a, b), c)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_FMulFAddOp : Op<NeuraDialect, "fmul_fadd"> {
  let summary = "Fused fadd(fmul(a, b), c)";
  let arguments = (ins AnyType:$a, AnyType:$b, AnyType:$c);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$a `,` $b `,` $c `,` attr-dict `:` type($result)";
  let traits = [SameOperandsAndResultElementType];
}

def Neura_MulAddOp : Op<NeuraDialect, "mul_add"> {
  let summary = "Fused add(mul(a, b), c)";
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

// Defines phi operation for merging values in dataflow form.
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

  // Explicitly specifies types for operands in the assembly format.
  // let assemblyFormat = "$init_val `:` type($init_val) `,` $loop_val `:` type($loop_val) attr-dict `,` type($result)";
}

// Defines control movement extending base move but with different signature.
def Neura_CtrlMovOp : Op<NeuraDialect, "ctrl_mov"> {
  let summary = "Control movement operation";
  let description = [{
    Connects a value to a reserved placeholder in the dataflow.
    Used to establish control flow dependencies.

    Example:
      ctrl_mov %value to %placeholder : f32  // Connect value to placeholder
  }];

  // Adds type constraints for both operands.
  let arguments = (ins AnyType:$value, AnyType:$target);
  let results = (outs);

  // Corrects assembly format - types must be space-separated.
  let assemblyFormat = "$value `->` $target attr-dict `:` type($value) type($target)";
}

// Defines reserve operation for control flow values.
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
      // Shows loop control that calculates next index and validity in one step.
      %next_idx, %loop_valid = neura.loop_control(
        parent_valid = %parent_valid,
      ) <{
           start_value = 0 : i64, 
           end_value = 100 : i64, 
           step_value = 1 : i64,
           iteration_type = "increment"
         }> : !neura.data<i1, i1> -> !neura.data<i64, i1>, !neura.data<i1, i1>
  }];

  let arguments = (ins
    AnyType:$parentValid,  // Valid predicate from the parent loop.
    AnyAttr:$iterationType, // "increment" or "decrement".
    AnyAttr:$start,       // Start index value.
    AnyAttr:$end,         // End index value.
    AnyAttr:$step         // Step value (increment or decrement based on iteration_type).
  );
  

  let results = (outs
    AnyType:$nextindex,         // Current loop index
    AnyType:$valid          // Valid predicate for the current index
  );

  // let assemblyFormat =
  //   " `(``parent_valid` `=` $parentValid `,` `start` `=` $start `,` `end` `=` $end `,` `step` `=` $step`)` attr-dict `:` type($parentValid) `,` type($start) `,` type($end) `,` type($step) `->` type($nextindex) `,` type($valid)";
}

// ----------------------------------------------------
// Defines operations for steering-control based DFG execution.
// ----------------------------------------------------

// Defines the true_steer operation.
def Neura_TrueSteerOp : Op<NeuraDialect, "true_steer">{
  let summary = "Conditionally pass a value when condition is true.";
  let description = [{When the condition is true, the input value is passed to the output; otherwise, the output is empty.
    Example:
      %out = neura.true_steer %in, %cond : i32, i1 -> f32
  }];

  let arguments = (ins AnyType:$input, AnyType:$condition);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input `,` $condition attr-dict `:` type($input) `,` type($condition) `->` type($output)";
}

// Defines the false_steer operation.
def Neura_FalseSteerOp : Op<NeuraDialect, "false_steer">{
  let summary = "Conditionally pass a value when condition is false.";
  let description = [{When the condition is false, the input value is passed to the output; otherwise, the output is empty.
    Example:
      %out = neura.false_steer %in, %cond : i32, i1 -> f32
  }];

  let arguments = (ins AnyType:$input, AnyType:$condition);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input `,` $condition attr-dict `:` type($input) `,` type($condition) `->` type($output)";
}

// Defines the carry operation.
def Neura_CarryOp : Op<NeuraDialect, "carry">{
  let summary = "Carry state across iterations.";
  let description = [{
  Three inputs for carry operation:
    - initial value: used in the first execution.
    - condition: determines whether to use the carried value.
    - carried value: used when condition is true.
  The output is the initial value when it is executed for the first time, otherwise it is the carried value when the condition is true.
  Example:
    %out = neura.carry %init, %cond, %carry_val : i32, i1, i32 -> i32
  }];

  let arguments = (ins AnyType:$initial, AnyType:$condition, AnyType:$carried);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$initial `,` $condition `,` $carried attr-dict `:` type($initial) `,` type($condition) `,` type($carried) `->` type($result)";
}

// Defines the merge operation.
def Neura_MergeOp : Op<NeuraDialect, "merge">{
  let summary = "Merge multiple inputs into one output.";
  let description = [{
  Merges multiple input values into a single output value based on the condition.
  Example:
      %out = neura.merge %cond, %in1, %in2 : i1, i32, i32 -> i32
  }];

  let arguments = (ins AnyType:$condition, AnyType:$true_value, AnyType:$false_value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$condition `,` $true_value `,` $false_value attr-dict `:` type($condition) `,` type($true_value) `,` type($false_value) `->` type($result)";
}

// Defines the invariant operation.
def Neura_InvariantOp : Op<NeuraDialect, "invariant">{
  let summary = "Invariant value across DFG execution.";
  let description = [{
  Invariant operation is a subset of carry operation where the output is always the initial value.
  Example:
      %out = neura.invariant %init %cond : i32, i1 -> i32
  }];
  let arguments = (ins AnyType:$initial, AnyType:$condition);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$initial `,` $condition attr-dict `:` type($initial) `,` type($condition) `->` type($result)";
}