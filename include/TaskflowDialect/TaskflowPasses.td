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

<<<<<<< HEAD
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
=======


def PlaceACTOnCGRA : Pass<"place-act-on-cgra", "func::FuncOp"> {
  let summary = "Places ACTs onto a 2D CGRA grid with adjacency optimization";
  let description = [{
    This pass places Atomic Canonical Tasks (ACTs) onto a 2D CGRA grid.
    Fusion candidates (same-header SSA dependencies) are placed on adjacent
    CGRAs to enable direct data forwarding.

    Uses a default 4x4 CGRA grid.
  }];
  let constructor = "taskflow::createPlaceACTOnCGRAPass()";
>>>>>>> a0f7fc7 (feat(taskflow): implement graph-based ACT placement and memory management framework)
}
#endif // TASKFLOW_PASSES_TD