// RUN: mlir-neura-opt %s \
// RUN: --promote-func-arg-to-const \
// RUN: --fold-constant \
// RUN: | FileCheck %s -check-prefix=FOLD

module {
  func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
    %0 = "neura.constant"() <{value = "%arg0"}> : () -> memref<?xi32>
    %1 = "neura.constant"() <{value = "%arg1"}> : () -> memref<?xi32>
    %2 = "neura.constant"() <{value = 1 : i64}> : () -> i64
    %3 = "neura.constant"() <{value = 128 : i64}> : () -> i64
    %4 = "neura.constant"() <{value = 1 : i32}> : () -> i32
    %5 = "neura.constant"() <{value = 2 : i32}> : () -> i32
    %6 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %7 = "neura.constant"() <{value = 2.5 : f32}> : () -> f32
    %8 = "neura.constant"() <{value = 1.0 : f32}> : () -> f32
    %9 = "neura.constant"() <{value = 0.0 : f32}> : () -> f32
    %10 = "neura.constant"() <{value = 10.0 : f32}> : () -> f32
    neura.br %6 : i64 to ^bb1
  ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb2
    %12 = "neura.icmp"(%11, %3) <{cmpType = "slt"}> : (i64, i64) -> i1
    neura.cond_br %12 : i1 then to ^bb2 else to ^bb3
  ^bb2:  // pred: ^bb1
    %13 = neura.load_indexed %0[%11 : i64] memref<?xi32> : i32
    %14 = "neura.mul"(%5, %13) : (i32, i32) -> i32
    %15 = "neura.add"(%4, %13) : (i32, i32) -> i32
    neura.store_indexed %15 to %1[%11 : i64] memref<?xi32> : i32
    
    // Test new float operations with constant folding
    %16 = "neura.cast"(%13) <{cast_type = "sitofp"}> : (i32) -> f32
    %17 = "neura.fmul"(%16, %7) : (f32, f32) -> f32
    %18 = "neura.fsub"(%17, %8) : (f32, f32) -> f32
    %19 = "neura.fmax"(%18, %9) : (f32, f32) -> f32
    %20 = "neura.fmin"(%19, %10) : (f32, f32) -> f32
    
    %21 = "neura.add"(%11, %2) : (i64, i64) -> i64
    neura.br %21 : i64 to ^bb1
  ^bb3:  // pred: ^bb1
    "neura.return"() : () -> ()
  }
}

// FOLD:      func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// FOLD-NEXT:   %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// FOLD-NEXT:   neura.br %0 : i64 to ^bb1
// FOLD-NEXT: ^bb1(%1: i64):  // 2 preds: ^bb0, ^bb2
// FOLD-NEXT:   %2 = "neura.icmp"(%1) <{cmpType = "slt"}> {rhs_value = 128 : i64} : (i64) -> i1
// FOLD-NEXT:   neura.cond_br %2 : i1 then to ^bb2 else to ^bb3
// FOLD-NEXT: ^bb2:  // pred: ^bb1
// FOLD-NEXT:   %3 = neura.load_indexed [%1 : i64] {lhs_value = "%arg0"} : i32
// FOLD-NEXT:   %4 = "neura.mul"(%3) {rhs_value = 2 : i32} : (i32) -> i32
// FOLD-NEXT:   %5 = "neura.add"(%3) {rhs_value = 1 : i32} : (i32) -> i32
// FOLD-NEXT:   neura.store_indexed %5 to [%1 : i64] {rhs_value = "%arg1"} : i32
// FOLD-NEXT:   %6 = "neura.cast"(%3) <{cast_type = "sitofp"}> : (i32) -> f32
// FOLD-NEXT:   %7 = "neura.fmul"(%6) {rhs_value = 2.500000e+00 : f32} : (f32) -> f32
// FOLD-NEXT:   %8 = "neura.fsub"(%7) {rhs_value = 1.000000e+00 : f32} : (f32) -> f32
// FOLD-NEXT:   %9 = neura.fmax<"maxnum"> (%8) {rhs_value = 0.000000e+00 : f32} : f32 -> f32
// FOLD-NEXT:   %10 = neura.fmin<"minnum"> (%9) {rhs_value = 1.000000e+01 : f32} : f32 -> f32
// FOLD-NEXT:   %11 = "neura.add"(%1) {rhs_value = 1 : i64} : (i64) -> i64
// FOLD-NEXT:   neura.br %11 : i64 to ^bb1
// FOLD-NEXT: ^bb3:  // pred: ^bb1
// FOLD-NEXT:   "neura.return"() : () -> ()
