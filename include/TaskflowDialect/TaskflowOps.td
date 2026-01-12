#ifndef TASKFLOW_OPS_TD
#define TASKFLOW_OPS_TD

include "TaskflowDialect/TaskflowDialect.td"
include "TaskflowDialect/TaskflowTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

//----------------------------------------------------------------------
// Base Class for all Taskflow operations.
//----------------------------------------------------------------------
class TaskflowOpBase<string mnemonic, list<Trait> traits = []> : Op<TaskflowDialect, mnemonic, traits>;

//----------------------------------------------------------------------
// Task Level Operations.
//----------------------------------------------------------------------

// Defines a uniform computation task operation within a Taskflow graph.
def TaskflowTaskOp : TaskflowOpBase<"task", [
    IsolatedFromAbove,
    AutomaticAllocationScope,
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    SingleBlockImplicitTerminator<"TaskflowYieldOp">
]>{
  let summary = "Computation task operation within a Taskflow graph.";

  let description = [{
    Represents a computational task that takes data inputs and produces
    data outoputs. Tasks are isolated from their surrounding scope and can only
    communicate through explicit data dependencies.

    Tasks has two types of inputs/outputs:
    1. Memory dependencies: memrefs that are read or written by the task
    2. Value dependencies: SSA values from producer tasks

    Example:
      // Memory input: %mem, Value input: %val
      $out_mem, %out_val = taskflow.task "Task_0"
        memory_inputs(%mem : memref<4xi32>)
        value_inputs(%val : i32) {
      ^bb0(%a0: memref<4xi32>, %a1: i32):
        affine.for %i = 0 to 4 {
          %v = affine.load %a0[%i] : memref<4xi32>
          %sum = arith.addi %v, %a1 : i32
          affine.store %sum, %a0[%i] : memref<4xi32>
        }
        taskflow.yield memory_outputs(%a0 : memref<4xi32>) value_outputs(%a1 : i32)
      } : (memref<4xi32>, i32) -> (memref<4xi32>, i32)
  }];

  let arguments = (ins
    Variadic<AnyMemRef>:$memory_inputs,
    Variadic<AnyType>:$value_inputs,
    StrAttr:$task_name
  );

  let results = (outs
    Variadic<AnyMemRef>:$memory_outputs,
    Variadic<AnyType>:$value_outputs
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`memory_inputs` `(` $memory_inputs^ `:` type($memory_inputs) `)`)?
    (`value_inputs` `(` $value_inputs^ `:` type($value_inputs) `)`)?
    attr-dict-with-keyword
    $body
    `->` `(`type($memory_outputs) type($value_outputs)`)`
  }];
}

// Defines the yield operation to terminate a Taskflow task.
def TaskflowYieldOp : TaskflowOpBase<"yield", [Terminator, Pure, ReturnLike, AttrSizedOperandSegments, ParentOneOf<["TaskflowTaskOp"]>]>{
  let summary = "Yield operation for Taskflow task";
  let description = [{
    Yields values from a task body. The number and types of operands
    must match the result types of the parent taskflow.task operation.
    
    Example:
      taskflow.task "Task_0" (%arg0, %arg1) {
        ...
        taskflow.yield %a0 : memref<4xi32>
      } : (memref<4xi32>, i32) -> memref<4xi32>
  }];

  let arguments = (ins
    Variadic<AnyMemRef>:$memory_results,
    Variadic<AnyType>:$value_results);

  let assemblyFormat = [{
    (`memory_outputs` `(` $memory_results^ `:` type($memory_results) `)`)?
    (`value_outputs` `(` $value_results^ `:` type($value_results) `)`)?
    attr-dict
  }];

  let builders = [
    // Default builder for empty yield.
    OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{}, ValueRange{});
    }]>
  ];
}

// Defines the data dependency edge operation that carries data dependencies between tasks in a Taskflow graph.
def TaskflowChannelOp : TaskflowOpBase<"channel", [Pure, SameOperandsAndResultType]>{
  let summary = "Data dependency edge that carries data dependencies between tasks in a Taskflow graph";
  let description = [{
    Represents a data dependency edge between tasks in the taskflow graph.
    A channel connects a producer task's output to a consumer task's input.

    Channels enforce explicit data dependencies and can be used for:
    - Producer-consumer relationships
    - Read-after-write (RAW) dependencies
    - Write-after-read (WAR) dependencies
    - Write-after-write (WAW) dependencies

    Example:
      %0 = taskflow.task "producer_task" (...) { ... } : (...) -> memref<4xi32>
      %1 = taskflow.channel %0 : memref<4xi32>
      %2 = taskflow.task "consumer_task" (%1, ...) { ... } : (memref<4xi32>, ...) -> ...
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$target);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($target)
  }];
}

#endif // TASKFLOW_OPS_TD