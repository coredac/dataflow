// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura
// RN:   --transform-ctrl-to-data-flow \
// RN:   %s | FileCheck %s

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
// CHECK-NEXT:   %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:   %1 = "neura.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
// CHECK-NEXT:   %2 = "neura.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> f32
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 4.000000e+00 : f32}> : () -> f32
// CHECK-NEXT:   %5 = "neura.icmp"(%arg0, %0) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT:   %6 = "neura.not"(%5) : (i1) -> i1
// CHECK-NEXT:   %7 = "neura.fmul"(%3, %4, %5) : (f32, f32, i1) -> f32
// CHECK-NEXT:   %8 = "neura.fadd"(%1, %2, %6) : (f32, f32, i1) -> f32
// CHECK-NEXT:   %9 = "neura.sel"(%7, %8, %5) : (f32, f32, i1) -> f32
// CHECK-NEXT:   return %9 : f32
// CHECK-NEXT: }