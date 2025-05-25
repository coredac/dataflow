// ConversionPasses.td - Passes for dialect conversion

#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Conversion passes
//=========================================================//
def LowerArithToNeura : Pass<"lower-arith-to-neura", "FuncOp">{
  let summary = "Lower arith to Neura dialect";
  let description = [{Lower arith dialect operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerArithToNeuraPass()";
}

def LowerLlvmToNeura : Pass<"lower-llvm-to-neura", "ModuleOp">{
  let summary = "Lower LLVM to Neura dialect";
  let description = [{Lower LLVM operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerLlvmToNeuraPass()";
}

#endif // CONVERSION_PASSES_TD