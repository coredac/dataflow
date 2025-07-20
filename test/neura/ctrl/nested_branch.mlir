// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

func.func @complex_test(%in: i64) -> f32 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1.0 : f32) : f32
  %c2 = llvm.mlir.constant(2.0 : f32) : f32
  %c3 = llvm.mlir.constant(3.0 : f32) : f32
  %c4 = llvm.mlir.constant(4.0 : f32) : f32
  %cond = llvm.icmp "eq" %in, %c0 : i64
  llvm.cond_br %cond, ^bb2, ^bb1(%c1 : f32)

^bb1(%true_val: f32):
  %loop_cond = llvm.fcmp "olt" %true_val, %c2 : f32
  llvm.cond_br %loop_cond, ^bb1_loop(%true_val : f32), ^bb3(%true_val : f32)

^bb1_loop(%loop_val: f32):
  %updated_val = llvm.fadd %loop_val, %c1 : f32
  llvm.br ^bb1(%updated_val : f32)

^bb2:
  %false_val = llvm.fmul %c3, %c4 : f32
  llvm.br ^bb3(%false_val : f32)

^bb3(%v: f32):
  return %v : f32
}

// CHECK:      func.func @complex_test(%arg0: i64) -> f32 attributes {accelerator = "neura"} {
// CHECK-NEXT:   %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %1 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %2 = "neura.constant"() <{predicate = true, value = 1.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %3 = "neura.constant"() <{predicate = true, value = 2.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %4 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %5 = "neura.constant"() <{predicate = true, value = 4.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %6 = "neura.icmp"(%0, %1) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %6 : !neura.data<i1, i1> then %4, %5 : !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb3 else %2, %3 : !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb1(%7: !neura.data<f32, i1>, %8: !neura.data<f32, i1>):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   %9 = "neura.fcmp"(%7, %8) <{cmpType = "olt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %9 : !neura.data<i1, i1> then %7, %2, %3 : !neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb2 else %7 : !neura.data<f32, i1> to ^bb4
// CHECK-NEXT: ^bb2(%10: !neura.data<f32, i1>, %11: !neura.data<f32, i1>, %12: !neura.data<f32, i1>):  // pred: ^bb1
// CHECK-NEXT:   %13 = "neura.fadd"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %13, %12 : !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb3(%14: !neura.data<f32, i1>, %15: !neura.data<f32, i1>):  // pred: ^bb0
// CHECK-NEXT:   %16 = "neura.fmul"(%14, %15) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %16 : !neura.data<f32, i1> to ^bb4
// CHECK-NEXT: ^bb4(%17: !neura.data<f32, i1>):  // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:   "neura.return"(%17) : (!neura.data<f32, i1>) -> ()
// CHECK-NEXT: }

// CTRL2DATA: func.func @complex_test(%arg0: i64) -> f32 attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 1.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_always"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %4 = "neura.grant_once"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 2.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.grant_always"(%5) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%5) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{predicate = true, value = 4.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.icmp"(%0, %1) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %14 = neura.grant_predicate %9, %13 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %15 = neura.grant_predicate %11, %13 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %16 = "neura.not"(%13) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %17 = neura.grant_predicate %4, %16 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %18 = neura.grant_predicate %7, %16 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %19 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %20 = "neura.phi"(%19, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %21 = neura.reserve : !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %22 = "neura.phi"(%21, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.fcmp"(%22, %20) <{cmpType = "olt"}> : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %24 = neura.grant_predicate %22, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %4, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %7, %23 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %27 = "neura.not"(%23) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %22, %27 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %29 = "neura.fadd"(%24, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %29 -> %21 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %26 -> %19 : !neura.data<f32, i1> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %30 = "neura.fmul"(%14, %15) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %31 = "neura.phi"(%28, %30) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%31) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT:   }