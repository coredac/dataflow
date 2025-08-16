// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   %s -o %t_dataflow.mlir 

// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt 2>&1

// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir

// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt

func.func @loop_sum() -> f32 {
  %init_i = arith.constant 0.0 : f32      
  %init_sum = arith.constant 0.0 : f32    
  %step = arith.constant 1.0 : f32        
  %add_val = arith.constant 3.0 : f32
  %limit = arith.constant 5.0 : f32       

  %v_i = "neura.reserve"() : () -> (f32)       
  %v_sum = "neura.reserve"() : () -> (f32)    

  "neura.ctrl_mov"(%init_i, %v_i) : (f32, f32) -> ()
  "neura.ctrl_mov"(%init_sum, %v_sum) : (f32, f32) -> ()

  "neura.br"() [^loop_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^loop_head:
  %i = "neura.phi"(%v_i, %init_i) : (f32, f32) -> f32       
  %sum = "neura.phi"(%v_sum, %init_sum) : (f32, f32) -> f32

  %cond = "neura.fcmp"(%i, %limit) {cmpType = "lt"} : (f32, f32) -> i1
  
  "neura.cond_br"(%cond) [^loop_body, ^loop_exit] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^loop_body:
  %new_sum = "neura.fadd"(%sum, %add_val) : (f32, f32) -> f32
  
  %new_i = "neura.fadd"(%i, %step) : (f32, f32) -> f32
  
  "neura.ctrl_mov"(%new_i, %v_i) : (f32, f32) -> ()
  "neura.ctrl_mov"(%new_sum, %v_sum) : (f32, f32) -> ()
  
  "neura.br"() [^loop_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^loop_exit:
  return %sum : f32
}

// DATAFLOW_IR: module {
// DATAFLOW_IR:   func.func @loop_sum() -> f32 attributes {accelerator = "neura"} {
// DATAFLOW_IR:     %cst = arith.constant 0.000000e+00 : f32
// DATAFLOW_IR:     %cst_0 = arith.constant 1.000000e+00 : f32
// DATAFLOW_IR:     %cst_1 = arith.constant 3.000000e+00 : f32
// DATAFLOW_IR:     %cst_2 = arith.constant 5.000000e+00 : f32
// DATAFLOW_IR:     %0 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR:     %1 = "neura.grant_once"(%0) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %2 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR:     %3 = "neura.grant_once"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %cst -> %0 : f32 !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %cst -> %2 : f32 !neura.data<f32, i1>
// DATAFLOW_IR:     %4 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR:     %5 = "neura.phi"(%4, %cst_2) : (!neura.data<f32, i1>, f32) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %6 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR:     %7 = "neura.phi"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %8 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR:     %9 = "neura.phi"(%8, %cst) : (!neura.data<f32, i1>, f32) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %10 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR:     %11 = "neura.phi"(%10, %1) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %12 = "neura.phi"(%11, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %13 = "neura.phi"(%7, %9) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %14 = "neura.fcmp"(%12, %5) <{cmpType = "lt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR:     %15 = neura.grant_predicate %13, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR:     %16 = neura.grant_predicate %cst_1, %14 : f32, !neura.data<i1, i1> -> f32
// DATAFLOW_IR:     %17 = neura.grant_predicate %12, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR:     %18 = neura.grant_predicate %cst_0, %14 : f32, !neura.data<i1, i1> -> f32
// DATAFLOW_IR:     %19 = neura.grant_predicate %1, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR:     %20 = neura.grant_predicate %3, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR:     %21 = neura.grant_predicate %cst, %14 : f32, !neura.data<i1, i1> -> f32
// DATAFLOW_IR:     %22 = neura.grant_predicate %cst_2, %14 : f32, !neura.data<i1, i1> -> f32
// DATAFLOW_IR:     %23 = "neura.not"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR:     %24 = neura.grant_predicate %13, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR:     %25 = "neura.fadd"(%15, %16) : (!neura.data<f32, i1>, f32) -> !neura.data<f32, i1>
// DATAFLOW_IR:     %26 = "neura.fadd"(%17, %18) : (!neura.data<f32, i1>, f32) -> !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %26 -> %19 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %25 -> %20 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %19 -> %10 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %21 -> %8 : f32 !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %20 -> %6 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR:     neura.ctrl_mov %22 -> %4 : f32 !neura.data<f32, i1>
// DATAFLOW_IR:     "neura.return"(%24) : (!neura.data<f32, i1>) -> ()
// DATAFLOW_IR:   }
// DATAFLOW_IR: }

// INTERPRETER_OUTPUT: [neura-interpreter]  Executing neura.return:
// INTERPRETER_OUTPUT: [neura-interpreter]  ├─ Return values:
// INTERPRETER_OUTPUT: [neura-interpreter]  │  └─15.000000, [pred = 1]
// INTERPRETER_OUTPUT: [neura-interpreter]  └─ Execution terminated successfully