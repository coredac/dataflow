// CompilerPasses.td - Passes for neura compiler

#ifndef COMPILER_PASSES_TD
#define COMPILER_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Passes for the CGRA Mapping
//=========================================================//
def GenerateDFG: Pass<"generate-dfg", "ModuleOp">{
  let summary = "Generates a Data Flow Graph (DFG) for the Neura dialect";
  let description = [{This pass generates a DFG from the Neura dialect operations.}];
  let constructor = "neura::createGenerateDFGPass()";
}

#endif // COMPILER_PASSES_TD