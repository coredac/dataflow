// ConversionPasses.td - Passes for dialect conversion

#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Neura Conversion Passes.
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

def LowerMemRefToNeura : Pass<"lower-memref-to-neura", "ModuleOp">{
  let summary = "Lower MemRef to Neura dialect";
  let description = [{Lower MemRef operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerMemRefToNeuraPass()";
}

def LowerBuiltinToNeura : Pass<"lower-builtin-to-neura", "ModuleOp">{
  let summary = "Lower Builtin to Neura dialect";
  let description = [{Lower Builtin operations to Neura dialect operations.}];
  let constructor = "mlir::createLowerBuiltinToNeuraPass()";
}

def LowerAffineToNeura : Pass<"lower-affine-to-neura", "func::FuncOp">{
  let summary = "Lower Affine perfect nested loops to Neura loop_control operations";
  let description = [{
    Converts perfectly nested affine.for loops directly to Neura dialect using 
    loop_control operations, avoiding the need to flatten to LLVM IR first.
    This preserves loop structure information for better optimization on 
    dataflow architectures.
  }];
  let constructor = "mlir::createLowerAffineToNeuraPass()";
  let dependentDialects = ["mlir::neura::NeuraDialect", "mlir::affine::AffineDialect"];
}

//=========================================================//
// TaskFlow Conversion Passes.
//=========================================================//
def ConvertLinalgToTaskflow : Pass<"convert-linalg-to-taskflow", "ModuleOp">{
  let summary = "Convert Linalg operations to Taskflow dialect";
  let description = [{
    Extracts compute-intensive linalg operations and wraps them into a Taskflow graph for spatial architecture execution.
  }];
  let constructor = "mlir::createConvertLinalgToTaskflowPass()";
  let dependentDialects = [
    "mlir::taskflow::TaskflowDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // CONVERSION_PASSES_TD