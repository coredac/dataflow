#ifndef TASKFLOW_OPS_TD
#define TASKFLOW_OPS_TD

include "TaskflowDialect/TaskflowDialect.td"
include "TaskflowDialect/TaskflowTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

// Defines a uniform computation and control task operation within a Taskflow graph.
def TaskflowTaskOp : TaskflowOpBase<"task", [
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    SingleBlockImplicitTerminator<"TaskflowYieldOp">,
    NoMemoryEffect,
]>{
  let summary = "Uniform computation and control task operation within a Taskflow graph";

  let description = [{
  
  }];

  let arguments = (ins
    // Optional control inputs.
    Variadic<TaskflowPacketType>:$control_ins,
    // Optional data inputs.
    Variadic<AnyType>:$data_ins,
    StrAttr:$task_name,

    // Task metadata.
    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
    OptionalAttr<ArrayAttr>:$iterator_types
  );

  let results = (outs
    // Optional control outputs.
    Variadic<TaskflowPacketType>:$control_outs,
    // Optional data outputs.
    Variadic<AnyType>:$data_outs
  );

  let regions = (region SizedRegion<1>:$body);

  // let assemblyFormat = [{
  //   $task_name
  //   (`control_ins` `(` $control_ins^ `:` type($control_ins) `)`)?
  //   (`data_ins` `(` $data_ins^ `:` type($data_ins) `)`)?
  //   $body attr-dict
  //   `->` type(results)
  // }];
}

// Defines the yield operation to terminate a Taskflow task.
def TaskflowYieldOp : TaskflowOpBase<"yield", [Terminator, Pure, ReturnLike, ParentOneOf<["TaskflowTaskOp"]>]>{
  let summary = "Yield operation for Taskflow task";
  let description = [{
  
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

// Defines the control edge operation that carries control packets between tasks in a Taskflow graph.
def TaskflowDriveOp : TaskflowOpBase<"drive", [Pure]>{
  let summary = "Control edge that carries control packets between tasks in a Taskflow graph";
  let description = [{
  
  }];

  let arguments = (ins TaskflowPacketType:$source);

  let results = (outs TaskflowPacketType:$target);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($target)
  }];
}

def TaskflowChannelOp : TaskflowOpBase<"channel", [Pure]>{
  let summary = "Data dependency edge that carries data dependencies between tasks in a Taskflow graph";
  let description = [{
  
  }];

  let arguments = (ins AnyType:$source);

  let results = (outs AnyType:$target);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($target)
  }];
}

#endif // TASKFLOW_OPS_TD