// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
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
// CHECK-NEXT: neura.cond_br %6 : i1 then :  to ^bb2 else :  to ^bb6
// CHECK-NEXT: ^bb2:  // pred: ^bb1
// CHECK-NEXT: %7 = "neura.cast"(%2) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT: neura.br %7 : i64 to ^bb3
// CHECK-NEXT: ^bb3(%8: i64):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT: %9 = "neura.cast"(%8) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT: %10 = "neura.icmp"(%9, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT: neura.cond_br %10 : i1 then :  to ^bb4 else :  to ^bb5
// CHECK-NEXT: ^bb4:  // pred: ^bb3
// CHECK-NEXT: %11 = neura.load_indexed %arg0[%2, %2, %2, %2, %2, %9] memref<?x1x1x1x1x128xi8> : i8
// CHECK-NEXT: neura.store_indexed %11 to %arg1[%2, %2, %5, %2, %2, %9] memref<?x1x128x1x1x128xi8> : i8
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
