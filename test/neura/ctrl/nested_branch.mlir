// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   %s | FileCheck %s

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
// CHECK-NEXT: %0 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
// CHECK-NEXT: %1 = "neura.constant"() <{predicate = true, value = 1.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %2 = "neura.constant"() <{predicate = true, value = 2.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %3 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %4 = "neura.constant"() <{predicate = true, value = 4.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %5 = "neura.icmp"(%arg0, %0) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT: %6 = "neura.not"(%5) : (i1) -> i1
// CHECK-NEXT: %7 = neura.grant_predicate %1, %6 : f32, i1 -> f32
// CHECK-NEXT: %8 = neura.reserve : f32
// CHECK-NEXT: %9 = "neura.phi"(%8, %7) : (f32, f32) -> f32
// CHECK-NEXT: %10 = "neura.fcmp"(%9, %2) <{cmpType = "olt"}> : (f32, f32) -> i1
// CHECK-NEXT: %11 = neura.grant_predicate %9, %10 : f32, i1 -> f32
// CHECK-NEXT: %12 = "neura.not"(%10) : (i1) -> i1
// CHECK-NEXT: %13 = neura.grant_predicate %9, %12 : f32, i1 -> f32
// CHECK-NEXT: %14 = "neura.fadd"(%11, %1) : (f32, f32) -> f32
// CHECK-NEXT: neura.ctrl_mov %14 -> %8 : f32 f32
// CHECK-NEXT: %15 = neura.grant_predicate %3, %5 : f32, i1 -> f32
// CHECK-NEXT: %16 = neura.grant_predicate %4, %5 : f32, i1 -> f32
// CHECK-NEXT: %17 = "neura.fmul"(%15, %16) : (f32, f32) -> f32
// CHECK-NEXT: %18 = "neura.phi"(%13, %17) : (f32, f32) -> f32
// CHECK-NEXT: "neura.return"(%18) : (f32) -> ()
// CHECK-NEXT: }