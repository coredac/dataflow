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

def FuseHyperblockToPipelinableTask : Pass<"fuse-hyperblock-to-pipelinable-task", "func::FuncOp"> {
  let summary = "Fuses hyperblocks into pipelinable tasks";
  let description = [{
    This pass fuses hyperblock into pipelinable tasks by analyzing data dependencies
    and segments memory access dependencies into different tasks.

    Conservative hyperblock fusion: split hyperblocks with memory access dependencies into different tasks,
    ensure each task has exactly one hyperblock"
  }];
  let constructor = "taskflow::createFuseHyperblockToPipelinableTaskPass()";
}
#endif // TASKFLOW_PASSES_TD