// TaskflowPasses.td - Passes for the Taskflow dialect

#ifndef TASKFLOW_PASSES_TD
#define TASKFLOW_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Passes for Task Level Optimizations
//=========================================================//
def AffineLoopTreeSerialization : Pass<"affine-loop-tree-serialization", "ModuleOp">{
  let summary = "Serializes top-level affine.for loops into minimized task operations";
  let description = [{
    This pass converts top-level affine.for loops in a function into
    minimized and canonicalized task operations.
  }];
  let constructor = "taskflow::createAffineLoopTreeSerializationPass()";
  let dependentDialects = [
    "mlir::taskflow::TaskflowDialect",
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect"];
}

//=========================================================//
// Passes for the Taskflow dialect
//=========================================================//
def ConstructHyperblockFromTask : Pass<"construct-hyperblock-from-task", "func::FuncOp">{
  let summary = "Constructs hyperblocks from Taskflow tasks by detecting perfect nested loop bands";
  let description = [{
    This pass constructs hyperblocks from Taskflow tasks by detecting perfect nested loop bands.
  }];
  let constructor = "taskflow::createConstructHyperblockFromTaskPass()";
}

def ClassifyCounters : Pass<"classify-counters", "ModuleOp">{
  let summary = "Classifies counters as root/relay/leaf";
  let description = [{
    Analyzes the counter hierarchy within taskflow.task operations and
    classifies each counter:
    - root: Top-level counter with no parent
    - relay: Intermediate counter with both parent and child counters
    - leaf: Innermost counter with no child counters

    Leaf counters are mapped to CGRA tile arrays.
  }];
  let constructor = "taskflow::createClassifyCountersPass()";
}

def MapTaskOnCgra : Pass<"map-task-on-cgra", "func::FuncOp"> {
  let summary = "Maps Taskflow tasks onto a 2D CGRA grid array";
  let description = [{
    This pass maps Taskflow tasks onto a 2D CGRA grid array.
    Fusion candidates (same-header SSA dependencies) are placed on adjacent
    CGRAs to enable direct data forwarding.

    Uses a default 3x3 CGRA grid.
  }];
  let constructor = "taskflow::createMapTaskOnCgraPass()";
}
#endif // TASKFLOW_PASSES_TD