// TaskflowPasses.td - Passes for the Taskflow dialect

#ifndef TASKFLOW_PASSES_TD
#define TASKFLOW_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Passes for Task Level Optimizations
//=========================================================//
def AffineLoopTreeSerialization
    : Pass<"affine-loop-tree-serialization", "ModuleOp"> {
  let summary =
      "Serializes top-level affine.for loops into minimized task operations";
  let description = [{
    This pass converts top-level affine.for loops in a function into
    minimized and canonicalized task operations.
  }];
  let constructor = "taskflow::createAffineLoopTreeSerializationPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}

def AffineLoopPerfection : Pass<"affine-loop-perfection", "func::FuncOp"> {
  let summary =
      "Perfectionizes affine.for loops into perfect nested loop bands";
  let description = [{
    This pass transforms affine.for loops into perfect nested loop bands by
    applying loop transformations such as loop fusion, loop interchange, and
    loop tiling.
  }];
  let constructor = "taskflow::createAffineLoopPerfectionPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}

//=========================================================//
// Passes for the Taskflow dialect
//=========================================================//
def ConstructHyperblockFromTask
    : Pass<"construct-hyperblock-from-task", "func::FuncOp"> {
  let summary = "Constructs hyperblocks from Taskflow tasks.";
  let description = [{
    This passes selected two transform modes based on the input architecture specification.
    1. With counter support: This pass constructs hyperblocks from Taskflow tasks by detecting perfect nested loop bands.
    2. Without counter support: This pass constructs hyperblocks from Taskflow tasks by wrapping the innermost loop as a hyperblock.
  }];
  let constructor = "taskflow::createConstructHyperblockFromTaskPass()";
}

def ClassifyCounters : Pass<"classify-counters", "ModuleOp"> {
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

def FuseTask : Pass<"fuse-task", "func::FuncOp"> {
  let summary =
      "Fuses Taskflow tasks using producer-consumer and sibling strategies";
  let description = [{
    Fuses taskflow.task operations using producer-consumer and sibling
    fusion strategies. Uses Neura-level MII metrics for profitability analysis.
    
    Producer-Consumer Fusion: Fuses a producer task into its consumer when
    the producer's memory output feeds directly into the consumer.
    
    Sibling Fusion: Fuses tasks that share inputs without data dependency.
  }];
  let constructor = "taskflow::createFuseTaskPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def TaskCategorization : Pass<"task-categorization", "func::FuncOp"> {
  let summary = "Categorizes tasks as divisible or atomic based on loop parallelism";
  let description = [{
    Analyzes each taskflow.task operation to determine whether its loop nest
    contains parallel loops that can be tiled for data-level parallelism.

    Task categories:
    - divisible: The task has at least one parallel loop (no loop-carried
      dependencies) whose trip count > 1.  Such tasks can be tiled into
      sibling sub-tasks for runtime configuration duplication (DLP).
    - atomic: The task has no exploitable parallel loops.  It must execute
      as a single indivisible unit.

    The pass attaches three attributes to each taskflow.task:
    - task_category : StringAttr  ("divisible" or "atomic")
    - parallel_dims : DenseI64ArrayAttr  (loop depth indices of parallel loops)
    - parallel_space : DenseI64ArrayAttr  (trip counts of the parallel dims)

    Parallel loop detection uses MLIR's affine dependence analysis
    (isLoopParallel).  Reduction loops recognised by MLIR are also counted
    as parallel.
  }];
  let constructor = "taskflow::createTaskCategorizationPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}

def MemoryAccessStreamingFusion
    : Pass<"memory-access-streaming-fusion", "func::FuncOp"> {
  let summary =
      "Fuses tasks connected by memory dependencies for streaming execution";
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

def ResourceAwareTaskOptimization
    : Pass<"resource-aware-task-optimization", "func::FuncOp"> {
  let summary = "Balances pipeline latency and fuses independent tasks for "
                "CGRA utilization.";
  let description = [{
    Two-phase optimization:
    1. Utilization Fusion: merges independent (no-edge) tasks, selecting pairs
       that minimize |trip_count_a - trip_count_b| for balanced utilization.
    2. Pipeline Balance: allocates extra CGRAs to critical-path bottleneck tasks.
       More CGRAs combine tile arrays into larger arrays for mapping, potentially
       lowering compiled_ii.  Latency model: II * (trip_count - 1) + steps.
    Targets a 4x4 CGRA grid (16 CGRAs total, one CGRA per cell).
    Currently a single task may be allocated at most 4 CGRAs.
    Supported CGRA array shapes for a task (all fit within the 4×4 grid):
      - rect : a perfect rectangle, e.g. 1×1, 1×2, 2×1, 1×3, 3×1, 2×2, 1×4, 4×1.
      - L    : an L-shaped block of 3 or 4 CGRAs, e.g.
                 3 CGRAs: (0,0)(1,0)(0,1) — two in a row + one below-left.
                 4 CGRAs: (0,0)(0,1)(0,2)(1,2) — three in a column + one offset.
      - T    : a T-shaped block of 4 CGRAs, e.g.
                 (0,0)(1,0)(2,0)(1,1) — three in a row + one below centre.
    Non-rectangular shapes are represented by their bounding box plus an
    explicit tile list that enumerates only the occupied CGRA positions.
    Compiled_ii must come from the downstream Neura pipeline (asserts on failure).

    Use --estimation-mode to control how task II/steps are estimated:
      compiled   (default): runs the full Neura lowering + mapping pipeline
                  for accurate compiled_ii and steps.
      analytical : uses only ResMII/RecMII analytical estimates without
                  running the mapper — much faster but less accurate.
                  Useful for rapid design-space exploration.
    Example:
      --resource-aware-task-optimization estimation-mode=analytical
  }];
  let options = [Option<
      "estimationMode", "estimation-mode", "std::string",
      /*default=*/"\"compiled\"",
      "Profiling estimation mode: 'compiled' (default) runs the full "
      "Neura lowering + mapping pipeline; 'analytical' uses only "
      "ResMII/RecMII analytical estimates (faster but less accurate).">];
  let constructor = "taskflow::createResourceAwareTaskOptimizationPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}
#endif // TASKFLOW_PASSES_TD