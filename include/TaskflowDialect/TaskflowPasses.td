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

def FuseTask : Pass<"fuse-task", "func::FuncOp"> {
  let summary = "Fuses Taskflow tasks using producer-consumer and sibling strategies";
  let description = [{
    Fuses taskflow.task operations using producer-consumer and sibling
    fusion strategies. Uses Neura-level MII metrics for profitability analysis.
    
    Producer-Consumer Fusion: Fuses a producer task into its consumer when
    the producer's memory output feeds directly into the consumer.
    
    Sibling Fusion: Fuses tasks that share inputs without data dependency.
  }];
  let constructor = "taskflow::createFuseTaskPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::neura::NeuraDialect",
    "mlir::taskflow::TaskflowDialect"];
}
#endif // TASKFLOW_PASSES_TD