// RUN: mlir-neura-opt %s \
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
    neura.br %6 : i64 to ^bb1
  ^bb1(%7: i64):  // 2 preds: ^bb0, ^bb2
    %8 = "neura.icmp"(%7, %3) <{cmpType = "slt"}> : (i64, i64) -> i1
    neura.cond_br %8 : i1 then to ^bb2 else to ^bb3
  ^bb2:  // pred: ^bb1
    %9 = neura.load_indexed %0[%7 : i64] memref<?xi32> : i32
    %10 = "neura.mul"(%5, %9) : (i32, i32) -> i32
    %11 = "neura.add"(%4, %9) : (i32, i32) -> i32
    neura.store_indexed %11 to %1[%7 : i64] memref<?xi32> : i32
    %12 = "neura.add"(%7, %2) : (i64, i64) -> i64
    neura.br %12 : i64 to ^bb1
  ^bb3:  // pred: ^bb1
    "neura.return"() : () -> ()
  }
}

// FOLD:      func.func @_Z11simple_loopPiS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// FOLD-NEXT:   %0 = "neura.constant"() <{value = "%arg0"}> : () -> memref<?xi32>
// FOLD-NEXT:   %1 = "neura.constant"() <{value = "%arg1"}> : () -> memref<?xi32>
// FOLD-NEXT:   %2 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// FOLD-NEXT:   neura.br %2 : i64 to ^bb1
// FOLD-NEXT: ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
// FOLD-NEXT:   %4 = "neura.icmp"(%3) <{cmpType = "slt"}> {rhs_value = 128 : i64} : (i64) -> i1
// FOLD-NEXT:   neura.cond_br %4 : i1 then to ^bb2 else to ^bb3
// FOLD-NEXT: ^bb2:  // pred: ^bb1
// FOLD-NEXT:   %5 = neura.load_indexed %0[%3 : i64] memref<?xi32> : i32
// FOLD-NEXT:   %6 = "neura.mul"(%5) {rhs_value = 2 : i32} : (i32) -> i32
// FOLD-NEXT:   %7 = "neura.add"(%5) {rhs_value = 1 : i32} : (i32) -> i32
// FOLD-NEXT:   neura.store_indexed %7 to %1[%3 : i64] memref<?xi32> : i32
// FOLD-NEXT:   %8 = "neura.add"(%3) {rhs_value = 1 : i64} : (i64) -> i64
// FOLD-NEXT:   neura.br %8 : i64 to ^bb1
// FOLD-NEXT: ^bb3:  // pred: ^bb1
// FOLD-NEXT:   "neura.return"() : () -> ()
