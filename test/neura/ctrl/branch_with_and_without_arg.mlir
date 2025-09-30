// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

func.func @test(%in: i64) -> f32 {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1.0 : f32) : f32
  %c2 = llvm.mlir.constant(2.0 : f32) : f32
  %c3 = llvm.mlir.constant(3.0 : f32) : f32
  %cond = llvm.icmp "eq" %in, %c0 : i64
  llvm.cond_br %cond, ^bb2(%c3 : f32), ^bb1(%c1, %c2 : f32, f32)

^bb1(%ca: f32, %cb: f32):
  %a = llvm.fadd %ca, %cb : f32
  llvm.br ^bb3(%a : f32)

^bb2(%cc: f32):
  %b = llvm.fmul %cc, %c2 : f32
  llvm.br ^bb3(%b : f32)

^bb3(%v: f32):
  return %v : f32
}

// CHECK: func.func @test(%arg0: i64) -> f32 attributes {accelerator = "neura"} {
// CHECK-NEXT:   %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %1 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %2 = "neura.constant"() <{value = 1.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 2.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %5 = "neura.icmp"(%0, %1) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %5 : !neura.data<i1, i1> then %4, %3 : !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb2 else %2, %3 : !neura.data<f32, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb1(%6: !neura.data<f32, i1>, %7: !neura.data<f32, i1>):  // pred: ^bb0
// CHECK-NEXT:   %8 = "neura.fadd"(%6, %7) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %8 : !neura.data<f32, i1> to ^bb3
// CHECK-NEXT: ^bb2(%9: !neura.data<f32, i1>, %10: !neura.data<f32, i1>):  // pred: ^bb0
// CHECK-NEXT:   %11 = "neura.fmul"(%9, %10) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %11 : !neura.data<f32, i1> to ^bb3
// CHECK-NEXT: ^bb3(%12: !neura.data<f32, i1>):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:   "neura.return"(%12) : (!neura.data<f32, i1>) -> ()
// CHECK-NEXT: }

// CTRL2DATA: func.func @test(%arg0: i64) -> f32 attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %1 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 1.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 2.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.icmp"(%0, %1) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %10 = neura.grant_predicate %7, %9 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %11 = neura.grant_predicate %5, %9 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.not"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %13 = neura.grant_predicate %3, %12 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %14 = neura.grant_predicate %5, %12 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %15 = "neura.fadd"(%13, %14) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %16 = "neura.fmul"(%10, %11) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     %17 = "neura.phi"(%15, %16) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CTRL2DATA-NEXT:     "neura.return"(%17) : (!neura.data<f32, i1>) -> ()
// CTRL2DATA-NEXT:   }