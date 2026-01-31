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

def ConvertAffineToTaskflow : Pass<"convert-affine-to-taskflow", "ModuleOp">{
  let summary = "Convert top-level affine.for operations to Taskflow dialect";
  let description = [{
    This pass converts top-level affine.for loops in a function into
    taskflow.task operations within a taskflow.graph. Each top-level loop
    becomes a separate task, and data dependencies between tasks are made
    explicit through taskflow.channel operations.
    
    The pass:
    1. Identifies all top-level affine.for operations
    2. Analyzes data dependencies (RAW, WAR, WAW) between loops
    3. Creates a taskflow.graph containing the loops
    4. Converts each loop to a taskflow.task
    5. Inserts taskflow.channel operations for data dependencies
  }];

  let constructor = "mlir::createConvertAffineToTaskflowPass()";
  let dependentDialects = [
    "mlir::taskflow::TaskflowDialect",
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def ConvertTaskflowToNeura : Pass<"convert-taskflow-to-neura", "ModuleOp">{
  let summary = "Convert taskflow.hyperblock to neura.kernel";
  let description = [{
    Converts taskflow.hyperblock operations with leaf counters into neura.kernel
    operations suitable for CGRA tile array mapping.
  }];
  let constructor = "mlir::createConvertTaskflowToNeuraPass()";
  let dependentDialects = [
    "mlir::taskflow::TaskflowDialect",
    "mlir::neura::NeuraDialect"
  ];
}

#endif // CONVERSION_PASSES_TD