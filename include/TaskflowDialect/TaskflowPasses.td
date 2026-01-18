// TaskflowPasses.td - Passes for the Taskflow dialect

#ifndef TASKFLOW_PASSES_TD
#define TASKFLOW_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Passes for the Taskflow dialect
//=========================================================//
def ConstructHyperblockFromTask : Pass<"construct-hyperblock-from-task", "func::FuncOp"> {
  let summary = "Constructs hyperblocks and counter chain from Taskflow tasks";
  let description = [{
    This pass constructs hyperblocks and counter chain from Taskflow tasks.
  }];
  let constructor = "taskflow::createConstructHyperblockFromTaskPass()";
}

def OptimizeTaskGraph : Pass<"optimize-task-graph", "func::FuncOp"> {
  let summary = "Optimizes Taskflow task graph by fusing hyperblocks and tasks.";
  let description = [{
    Performs the following optimizations on the Taskflow task graph:
    1. Hyperblock Fusion: Merges hyperblocks with compatible counter structures.
       Supports loop peeling when counter bound differences are small.
    2. Task Fusion: Merges producer-consumer tasks to reduce data transfer
       overhead between tasks.
    3. Dead Hyperblock Elimination: Removes unused hyperblocks.
  }];
  let constructor = "taskflow::createOptimizeTaskGraphPass()";
  let options = [
    Option<"enableHyperblockFusion", "enable-hyperblock-fusion", "bool",
           /*default=*/"true", "Enables hyperblock fusion optimization.">,
    Option<"enableTaskFusion", "enable-task-fusion", "bool",
           /*default=*/"true", "Enables task fusion optimization.">,
    Option<"maxBoundDiffForPeeling", "max-bound-diff", "int",
           /*default=*/"2", "Specifies max loop bound difference for peeling.">
  ];
}

#endif // TASKFLOW_PASSES_TD