#ifndef TASKFLOW_OPS_TD
#define TASKFLOW_OPS_TD

include "TaskflowDialect/TaskflowDialect.td"
include "TaskflowDialect/TaskflowTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//----------------------------------------------------------------------
// Base Class for all Taskflow operations.
//----------------------------------------------------------------------
class TaskflowOpBase<string mnemonic, list<Trait> traits = []> : Op<TaskflowDialect, mnemonic, traits>;

//----------------------------------------------------------------------
// Graph Level Operations.
//----------------------------------------------------------------------

// Defines the top-level graph operation representing the workload.
def TaskflowGraphOp : TaskflowOpBase<"graph", [
    IsolatedFromAbove,
    AutomaticAllocationScope,
    SingleBlockImplicitTerminator<"TaskflowReturnOp">
]>{
  let summary = "Top-level graph operation representing workload on a scale-out/scale-up spatial architecture.";  

  let description = [{
    Defines a region where all operations are flat tasks connected by edges.
    This is the boundary between Host (CPU) and Device (spatial architecture).
    
    The graph contains:
    - A flat list of `taskflow.task` operations (nodes)
    - `taskflow.drive` operations (control edges)
    - `taskflow.connect` operations (data dependency edges)
    - A single `taskflow.return` operation to terminate the graph.
    
    Example:

  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict-with-keyword $body `:` functional-type($inputs, $results)
  }];
}

// Defines the return operation to terminate a Taskflow graph.
def TaskflowReturnOp : TaskflowOpBase<"return", [Terminator]> {
  let summary = "Return operation for Taskflow graph.";
  let description = [{
    This operation terminates a Taskflow graph.
    This acts as a interaction op between the spatial architecture and the host processor.

    Example"
      taskflow.graph {
        ...
        taskflow.return
      }
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ($results^ `:` type($results))? attr-dict
  }];

  let builders = [
    // Default builder for empty return.
    OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
    }]>
  ];
}

//----------------------------------------------------------------------
// Task Level Operations.
//----------------------------------------------------------------------

// Defines a uniform computation task operation within a Taskflow graph.
def TaskflowTaskOp : TaskflowOpBase<"task", [
    IsolatedFromAbove,
    AutomaticAllocationScope,
    SingleBlockImplicitTerminator<"TaskflowYieldOp">
]>{
  let summary = "Computation task operation within a Taskflow graph.";

  let description = [{
    Represents a computational task that takes data inputs and produces
    data outoputs. Tasks are isolated from their surrounding scope and can only
    communicate through explicit data dependencies.

    Example:
      %result = taskflow.task "Task_0" (%arg0, %arg1) {
      ^bb0(%a0: memref<4xi32>, %a1: i32):
        affine.for %i = 0 to 4 {
          %v = affine.load %a0[%i] : memref<4xi32>
          %sum = arith.addi %v, %a1 : i32
          affine.store %sum, %a0[%i] : memref<4xi32>
        }
        taskflow.yield %a0 : memref<4xi32>
      } : (memref<4xi32>, i32) -> memref<4xi32>
  }];

  let arguments = (ins
    // Optional data inputs.
    Variadic<AnyType>:$inputs,
    StrAttr:$task_name
  );

  let results = (outs
    // Optional data outputs.
    Variadic<AnyType>:$outputs
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $task_name
    (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
    attr-dict-with-keyword
    $body
    `->` type($outputs)
  }];
}

// Defines the yield operation to terminate a Taskflow task.
def TaskflowYieldOp : TaskflowOpBase<"yield", [Terminator, Pure, ReturnLike, ParentOneOf<["TaskflowTaskOp"]>]>{
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

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ($results^ `:` type($results))? attr-dict
  }];

  let builders = [
    // Default builder for empty yield.
    OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
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