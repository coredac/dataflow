// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: | FileCheck %s

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s -check-prefix=CTRL2DATA
module attributes {} {
  func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {} {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 128 {
        %0 = affine.load %arg0[0, 0, 0, 0, 0, %arg3] : memref<?x1x1x1x1x128xi8>
        affine.store %0, %arg1[0, 0, %arg2, 0, 0, %arg3] : memref<?x1x128x1x1x128xi8>
      }
    }
    return
  }
}

// CHECK: func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura"} {
// CHECK-NEXT: %0 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT: %1 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> index
// CHECK-NEXT: %2 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
// CHECK-NEXT: %3 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %3 : i64 to ^bb1
// CHECK-NEXT: ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT: %5 = "neura.cast"(%4) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %6 = "neura.icmp"(%5, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %6 : i1 then to ^bb2 else to ^bb6
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %7 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %7 : i64 to ^bb3
// CHECK-NEXT: ^bb3(%8: i64):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT: %9 = "neura.cast"(%8) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %10 = "neura.icmp"(%9, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %10 : i1 then to ^bb4 else to ^bb5
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT: %11 = neura.load_indexed %arg0[%2, %2, %2, %2, %2, %9 : index, index, index, index, index, index] memref<?x1x1x1x1x128xi8> : i8
// CHECK-NEXT: neura.store_indexed %11 to %arg1[%2, %2, %5, %2, %2, %9 : index, index, index, index, index, index] memref<?x1x128x1x1x128xi8> : i8
// CHECK-NEXT: %12 = "neura.add"(%9, %0) : (index, index) -> index
// CHECK-NEXT: %13 = "neura.cast"(%12) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %13 : i64 to ^bb3
// CHECK-NEXT: ^bb5:  // pred: ^bb3
// CHECK-NEXT: %14 = "neura.add"(%5, %0) : (index, index) -> index
// CHECK-NEXT: %15 = "neura.cast"(%14) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %15 : i64 to ^bb1
// CHECK-NEXT: ^bb6:  // pred: ^bb1
// CHECK-NEXT: "neura.return"() : () -> ()
// CHECK-NEXT: }


// CTRL2DATA: func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) attributes {accelerator = "neura"} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x1x1x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x1x128x1x1x128xi8>, i1>) -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.phi"(%10, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.phi"(%12, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.icmp"(%13, %11) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %15 = "neura.not"(%14) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %16 = neura.grant_predicate %9, %14 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = "neura.phi"(%17, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = "neura.phi"(%19, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %21 = "neura.icmp"(%20, %18) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %22 = neura.grant_predicate %1, %21 : !neura.data<memref<?x1x1x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x1x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %23 = neura.grant_predicate %9, %21 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = neura.grant_predicate %20, %21 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %3, %21 : !neura.data<memref<?x1x128x1x1x128xi8>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x1x128x1x1x128xi8>, i1>
// CTRL2DATA-NEXT:     %26 = neura.grant_predicate %13, %21 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %5, %21 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %7, %21 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = "neura.not"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %13, %29 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.grant_predicate %5, %29 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %7, %29 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.load_indexed %22[%23, %23, %23, %23, %23, %24 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x1x1x1x1x128xi8>, i1> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %33 to %25[%23, %23, %26, %23, %23, %24 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x1x128x1x1x128xi8>, i1> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     %34 = "neura.add"(%24, %27) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %34 -> %19 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %28 -> %17 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = "neura.add"(%30, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %35 -> %12 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %32 -> %10 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }