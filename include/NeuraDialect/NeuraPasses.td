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

def CanonicalizeLiveIn : Pass<"canonicalize-live-in", "ModuleOp"> {
  let summary = "Canonicalizes live-in values/operations in each basic block.";
  let description = [{
    This pass applies canonicalization transformations to Neura dialect operations.
    The canonicalization includes:
    1. Converting all live-in values of each basic block to block arguments.
    2. Promoting function arguments to neura constant operations.
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

def InitCommonPattern : Pass<"init-common-pattern", "ModuleOp"> {
  let summary = "Initialize and identify common patterns in DFG";
  let description = [{
    This pass traverses the entire DFG (Data Flow Graph) and identifies
    common patterns using graph mining algorithms. Patterns that occur
    more than a threshold number of times are identified, printed, and
    stored in a data structure for future use.
  }];
  let constructor = "neura::createInitCommonPatternPass()";
}

#endif // NEURA_PASSES_TD