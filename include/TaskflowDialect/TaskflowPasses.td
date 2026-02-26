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

def AffineLoopPerfection : Pass<"affine-loop-perfection", "func::FuncOp">{
  let summary = "Perfectionizes affine.for loops into perfect nested loop bands";
  let description = [{
    This pass transforms affine.for loops into perfect nested loop bands by
    applying loop transformations such as loop fusion, loop interchange, and
    loop tiling.
  }];
  let constructor = "taskflow::createAffineLoopPerfectionPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect"];
}

//=========================================================//
// Passes for the Taskflow dialect
//=========================================================//
def ConstructHyperblockFromTask : Pass<"construct-hyperblock-from-task", "func::FuncOp">{
  let summary = "Constructs hyperblocks from Taskflow tasks.";
  let description = [{
    This passes selected two transform modes based on the input architecture specification.
    1. With counter support: This pass constructs hyperblocks from Taskflow tasks by detecting perfect nested loop bands.
    2. Without counter support: This pass constructs hyperblocks from Taskflow tasks by wrapping the innermost loop as a hyperblock.
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

def MemoryAccessStreamingFusion : Pass<"memory-access-streaming-fusion", "func::FuncOp"> {
  let summary = "Fuses tasks connected by memory dependencies for streaming execution";
  let description = [{
    Identifies and fuses taskflow.task operations that have memory access
    dependencies (one task writes a memref, another task reads it).
    Eliminates intermediate memref allocations and converts memory access
    dependencies into direct SSA value dependencies.
    
    Uses greedy fusion strategy to iteratively fuse the most beneficial
    task pairs until no more fusion opportunities exist.
  }];
  let constructor = "taskflow::createMemoryAccessStreamingFusionPass()";
}

def ResourceAwareTaskOptimization : Pass<"resource-aware-task-optimization", "func::FuncOp"> {
  let summary = "Balances pipeline latency and fuses independent tasks for CGRA utilization";
  let description = [{
    Two-phase optimization:
    1. Pipeline Balance: allocates extra CGRAs to critical-path bottleneck tasks
       to reduce their effective latency (trip_count / cgra_count).
    2. Utilization Fusion: merges independent (no-edge) tasks into a single task
       to reduce total CGRA count.
    Targets a 4x4 CGRA grid (16 CGRAs). Asserts if tasks exceed budget.
  }];
  let constructor = "taskflow::createResourceAwareTaskOptimizationPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect"];
}
#endif // TASKFLOW_PASSES_TD