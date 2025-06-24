// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA
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
// CHECK-NEXT: %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT: %2 = "neura.constant"() <{value = 0 : index}> : () -> index
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
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 128 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_always"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_always"(%4) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %6 = "neura.cast"(%4) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.phi"(%7, %8) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.cast"(%9) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %11 = "neura.icmp"(%10, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %12 = "neura.not"(%11) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %13 = "neura.cast"(%5) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.grant_predicate %13, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = "neura.phi"(%14, %15) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.cast"(%16) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %18 = "neura.icmp"(%17, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %19 = "neura.not"(%18) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %20 = neura.load_indexed %arg0[%5, %5, %5, %5, %5, %17 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x1x1x1x1x128xi8> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     %21 = neura.grant_predicate %20, %18 : !neura.data<i8, i1>, !neura.data<i1, i1> -> !neura.data<i8, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %21 to %arg1[%5, %5, %10, %5, %5, %17 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x1x128x1x1x128xi8> : !neura.data<i8, i1>
// CTRL2DATA-NEXT:     %22 = "neura.add"(%17, %1) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %23 = neura.grant_predicate %22, %18 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %24 = "neura.cast"(%23) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.grant_predicate %24, %18 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %25 -> %15 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = "neura.add"(%10, %1) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %26, %19 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %28 = "neura.cast"(%27) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %28, %19 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %29 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }