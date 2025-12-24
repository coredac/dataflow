// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   %s -o %t_dataflow.mlir 

// RUN: neura-interpreter %t_dataflow.mlir --verbose --dataflow > %t_output.txt

// RUN: FileCheck %s --check-prefix=DATAFLOW_IR --input-file=%t_dataflow.mlir

// RUN: FileCheck %s --check-prefix=INTERPRETER_OUTPUT --input-file=%t_output.txt

func.func @loop_sum() -> f32 {
  %c0 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> : () -> f32   // init_i / init_sum
  %c1 = "neura.constant"() <{predicate = true, value = 1.000000e+00 : f32}> : () -> f32   // step
  %c2 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> f32   // add_val
  %c3 = "neura.constant"() <{predicate = true, value = 5.000000e+00 : f32}> : () -> f32   // limit

  // Entry: jump to loop head with initial i and sum
  neura.br %c0, %c0 : f32, f32 to ^loop_head

^loop_head(%i: f32, %sum: f32):  // predecessors: entry, ^loop_body
  %cond = "neura.fcmp"(%i, %c3) <{cmpType = "lt"}> : (f32, f32) -> i1
  // If cond true, go to body carrying i and sum; else go to exit with sum
  neura.cond_br %cond : i1 then %i, %sum : f32, f32 to ^loop_body else %sum : f32 to ^loop_exit

^loop_body(%i_in: f32, %sum_in: f32):  // pred: ^loop_head
  %new_sum = "neura.fadd"(%sum_in, %c2) : (f32, f32) -> f32
  %new_i = "neura.fadd"(%i_in, %c1) : (f32, f32) -> f32
  // loop back with updated i and sum
  neura.br %new_i, %new_sum : f32, f32 to ^loop_head

^loop_exit(%ret_sum: f32):  // pred: ^loop_head
  "neura.return"(%ret_sum) : (f32) -> ()
}


// DATAFLOW_IR:        func.func @loop_sum() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate"} {
// DATAFLOW_IR-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %1 = "neura.grant_once"() <{constant_value = 1.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %2 = "neura.grant_once"() <{constant_value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %3 = "neura.grant_once"() <{constant_value = 5.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %4 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %5 = neura.phi_start %1, %4 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %6 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %7 = neura.phi_start %2, %6 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %8 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %9 = neura.phi_start %3, %8 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %10 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %11 = neura.phi_start %0, %10 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %12 = neura.reserve : !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %13 = neura.phi_start %0, %12 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %14 = "neura.fcmp"(%13, %9) <{cmpType = "lt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT:     %15 = neura.grant_predicate %13, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %16 = neura.grant_predicate %11, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %17 = neura.grant_predicate %7, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %18 = neura.grant_predicate %5, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %19 = neura.grant_predicate %9, %14 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %20 = "neura.not"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// DATAFLOW_IR-NEXT:     %21 = neura.grant_predicate %11, %20 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %22 = "neura.fadd"(%16, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     %23 = "neura.fadd"(%15, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     neura.ctrl_mov %23 -> %12 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     neura.ctrl_mov %22 -> %10 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     neura.ctrl_mov %19 -> %8 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     neura.ctrl_mov %17 -> %6 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     neura.ctrl_mov %18 -> %4 : !neura.data<f32, i1> !neura.data<f32, i1>
// DATAFLOW_IR-NEXT:     "neura.return"(%21) : (!neura.data<f32, i1>) -> ()
// DATAFLOW_IR-NEXT:   }

// INTERPRETER_OUTPUT: [neura-interpreter]  DFG Iteration 5 | Topological Level 6 | ready_to_execute_ops 3
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  ========================================
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  Executing operation: "neura.return"(%21) : (!neura.data<f32, i1>) -> ()
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  Executing neura.return:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  ├─ Return values:
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  │  └─15.000000, [pred = 1]
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  └─ Execution terminated successfully
// INTERPRETER_OUTPUT-NEXT: [neura-interpreter]  Termination signal received
