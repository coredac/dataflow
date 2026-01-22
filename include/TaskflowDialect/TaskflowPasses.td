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

def CanonicalizeTask: Pass<"canonicalize-task", "func::FuncOp">{
  let summary = "Canonicalizes tasks by splitting each hyperblock into a separate atomic task";
  let description = [{
    This pass splits tasks so that each task contains exactly one hyperblock.
    This creates atomic task units that can be analyzed and optimized independently.

    Input: Task with N hyperblocks
    Output: N atomic tasks, each containing one hyperblock

    This is a prerequisite pass before fusion optimizations.
  }];
  let constructor = "taskflow::createCanonicalizeTaskPass()";
}
#endif // TASKFLOW_PASSES_TD