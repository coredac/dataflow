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

def FusePatterns : Pass<"fuse-patterns", "ModuleOp"> {
  let summary = "Fuses patterns in the Neura dialect";
  let description = [{Apply Neura fusion patterns.}];
  let constructor = "neura::createFusePatternsPass()";
}

def InsertMov : Pass<"insert-mov", "ModuleOp"> {
  let summary = "Inserts move operations in the Neura dialect";
  let description =
      [{Insert neura.mov before and after all neura dialect operations.}];
  let constructor = "neura::createInsertMovPass()";
}

#endif // NEURA_PASSES_TD