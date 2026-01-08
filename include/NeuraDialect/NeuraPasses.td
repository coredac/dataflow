// NeuraPasses.td - Passes for the Neura dialect

#ifndef NEURA_PASSES_TD
#define NEURA_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Passes for the Neura dialect
//=========================================================//
def AssignAccelerator : Pass<"assign-accelerator", "ModuleOp"> {
  let summary = "Assigns accelerators to operations in the Neura dialect";
  let description = [{Tags non - main functions as neura.kernel.}];
  let constructor = "neura::createAssignAcceleratorPass()";
}

def FusePattern : Pass<"fuse-pattern", "ModuleOp"> {
  let summary = "Fuses patterns in the Neura dialect";
  let description = [{Apply Neura fusion patterns.}];
  let constructor = "neura::createFusePatternPass()";
}

def FuseKernel : Pass<"fuse-kernel", "ModuleOp"> {
  let summary = "Fuses kernel operations in the Neura dialect";
  let description = [{
    This pass fuses neura.kernel operations using producer-consumer and sibling
    fusion strategies, inspired by MLIR's linalg and affine loop fusion.
    
    Producer-Consumer Fusion: Fuses a producer kernel into its consumer when
    the producer's output is only used by the consumer.
    
    Sibling Fusion: Fuses kernels that share the same input operands and have
    no data dependencies between them.
  }];
  let constructor = "neura::createFuseKernelPass()";
}

def InsertDataMov : Pass<"insert-data-mov", "ModuleOp"> {
  let summary = "Inserts data move operations in the Neura dialect";
  let description =
      [{Insert neura.data_mov before all neura dialect operations.}];
  let constructor = "neura::createInsertDataMovPass()";
}

def InsertCtrlMov : Pass<"insert-ctrl-mov", "ModuleOp"> {
  let summary = "Inserts ctrl move operations in the Neura dialect";
  let description =
      [{Insert neura.ctrl_mov before all neura dialect operations.}];
  let constructor = "neura::createInsertCtrlMovPass()";
}

def TransformCtrlToDataFlow : Pass<"transform-ctrl-to-data-flow", "ModuleOp"> {
  let summary = "Inserts ctrl move operations in the Neura dialect";
  let description =
      [{Transform ctrl to predicate-based data flow.}];
  let constructor = "neura::createTransformCtrlToDataFlowPass()";
}

def LeveragePredicatedValue : Pass<"leverage-predicated-value", "ModuleOp"> {
  let summary = "Convert values to predicated values in Neura dialect";
  let description = [{
    This pass converts regular values to predicated values in Neura dialect operations.
    Each value is wrapped in a predicated value type with a default true predicate.
  }];
  let constructor = "neura::createLeveragePredicatedValuePass()";
}

def MapToAccelerator : Pass<"map-to-accelerator", "ModuleOp"> {
  let summary = "Map Neura operations onto a given accelerator";
  let description = [{
    This pass performs mapping from Neura operations to accelerator.
  }];
  let constructor = "neura::createMapToAcceleratorPass()";
}

def GenerateCode : Pass<"generate-code", "ModuleOp"> {
  let summary = "Generate JSON-formatted instructions from mapped Neura IR";
  let description = [{
    This pass generates JSON file containing the instructions.
    The instructions can be encoded into configuration signals.
  }];
  let constructor = "neura::createGenerateCodePass()";
}

def FuseLoopControl: Pass<"fuse-loop-control", "ModuleOp">{
  let summary = "Fuses loop control operations in the Neura dialect";
  let description = [{
    This pass fuses loop control operations.
  }];
  let constructor = "neura::createFuseLoopControlPass()";
}

def CanonicalizeReturn : Pass<"canonicalize-return", "ModuleOp">{
  let summary = "Adds execution conditions to void return operations";
  let description = [{
    This pass processes void functions and adds trigger conditions to their
    return operations. For void functions, the return operation needs an
    execution condition to properly trigger in dataflow mode.
    
    This pass should run before CanonicalizeLiveIn and TransformCtrlToDataFlow.
  }];
  let constructor = "neura::createCanonicalizeReturnPass()";
}

def CanonicalizeLiveIn : Pass<"canonicalize-live-in", "ModuleOp"> {
  let summary = "Canonicalizes live-in values/operations in each basic block.";
  let description = [{
    This pass applies canonicalization transformations to Neura dialect operations.
    The canonicalization includes:
    1. Converting all live-in values of each basic block to block arguments.
  }];
  let constructor = "neura::createCanonicalizeLiveInPass()";
}

def PromoteFuncArgToConst : Pass<"promote-func-arg-to-const", "ModuleOp"> {
  let summary = "Promotes function arguments to neura constant operations";
  let description = [{
    This pass promotes function arguments to neura constant operations.
  }];
  let constructor = "neura::createPromoteFuncArgToConstPass()";
}

def CanonicalizeCast : Pass<"canonicalize-cast", "ModuleOp"> {
  let summary = "Canonicalizes cast operations in the Neura dialect";
  let description = [{
    This pass applies canonicalization transformations to neura::cast operations.
    The canonicalization includes:
    1. Removing redundant casts.
    2. Converting index (i64) types to i64 (index).
  }];
  let constructor = "neura::createCanonicalizeCastPass()";
}

def FoldConstant : Pass<"fold-constant", "ModuleOp"> {
  let summary = "Folds constant operations in the Neura dialect";
  let description = [{
    This pass applies constant folding transformations to Neura dialect operations.
    The folding includes:
    1. Folding constant operations.
    2. Removing dead operations.
  }];
  let constructor = "neura::createFoldConstantPass()";
}

def TransformToSteerControl : Pass<"transform-to-steer-control", "func::FuncOp"> {
  let summary = "Transform control flow into data flow using steer control";
  let description = [{
    This pass transforms Neura control flow graphs (CDFG) into pure dataflow graphs (DFG)
    using steer control operations like true_steer, false_steer, carry, and merge.
    Unlike predication-based approaches, steer control explicitly directs data through
    different paths based on conditions.
  }];
  let constructor = "neura::createTransformToSteerControlPass()";
}

def RemovePredicatedType : Pass<"remove-predicated-type", "ModuleOp"> {
  let summary = "Removes predicated types from Neura dialect operations";
  let description = [{
    This pass removes predicated types from Neura dialect operations,
    converting them back to regular types.
  }];
  let constructor = "neura::createRemovePredicatedTypePass()";
}

def IterMergePattern : Pass<"iter-merge-pattern", "ModuleOp"> {
  let summary = "Iteratively merge and identify common patterns in DFG";
  let description = [{
    This pass iteratively traverses the DFG (Data Flow Graph) and identifies
    common patterns using graph mining algorithms. It performs iterative merging
    of patterns and stores results for subsequent passes.
  }];
  let constructor = "neura::createIterMergePatternPass()";
}

def InitPattern : Pass<"init-pattern", "ModuleOp"> {
  let summary = "Initialize and identify common patterns in DFG (single iteration)";
  let description = [{
    This pass traverses the DFG (Data Flow Graph) and identifies common patterns
    using graph mining algorithms. It performs a single iteration of pattern merging
    with a configurable minimum support threshold.
  }];
  let constructor = "neura::createInitPatternPass()";
}

def WrapLoopInKernelPass : Pass<"wrap-loop-in-kernel", "func::FuncOp">{
  let summary = "Wrap loops in neura.kernel operations";
  let description = [{
    This pass wraps loops in neura.kernel operations to encapsulate loop bodies.
  }];
  let constructor = "neura::createWrapLoopInKernelPass()";
}

def HardwareMerge : Pass<"hardware-merge", "ModuleOp"> {
  let summary = "Merge and optimize hardware units for pattern execution";
  let description = [{
    This pass analyzes patterns (fused_op regions) and designs an optimal
    hardware configuration that supports all patterns while minimizing
    hardware unit duplication. It considers the topological order of operations
    within each pattern to ensure correct hardware pipeline design.
    
    Algorithm:
    1. Extract pattern DAGs with topological structure from fused_op regions
    2. Group patterns by structural similarity (same DAG shape)
    3. For structurally similar patterns, create hardware templates where
       each slot supports all operations that appear at that position
    4. For structurally different patterns, use independent hardware units
    5. Output the final hardware configuration as a JSON file
  }];
  let constructor = "neura::createHardwareMergePass()";
}

def InitExecLatency : Pass<"init-exec-latency", "ModuleOp"> {
  let summary = "Initialize execution latency information";
  let description = [{
    This pass initializes execution latency information.
  }];
  let constructor = "neura::createInitExecLatencyPass()";
}
#endif // NEURA_PASSES_TD