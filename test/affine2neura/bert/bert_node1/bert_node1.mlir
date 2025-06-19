// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura | FileCheck %s
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
// CHECK-NEXT:    %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:    %1 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:    %2 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : index to i64
// CHECK-NEXT:    llvm.br ^bb1(%3 : i64)
// CHECK-NEXT:  ^bb1(%4: i64):  // 2 preds: ^bb0, ^bb5
// CHECK-NEXT:    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
// CHECK-NEXT:    %6 = "neura.icmp"(%5, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:    llvm.cond_br %6, ^bb2, ^bb6
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    %7 = builtin.unrealized_conversion_cast %2 : index to i64
// CHECK-NEXT:    llvm.br ^bb3(%7 : i64)
// CHECK-NEXT:  ^bb3(%8: i64):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:    %9 = builtin.unrealized_conversion_cast %8 : i64 to index
// CHECK-NEXT:    %10 = "neura.icmp"(%9, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:    llvm.cond_br %10, ^bb4, ^bb5
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    %11 = memref.load %arg0[%2, %2, %2, %2, %2, %9] : memref<?x1x1x1x1x128xi8>
// CHECK-NEXT:    memref.store %11, %arg1[%2, %2, %5, %2, %2, %9] : memref<?x1x128x1x1x128xi8>
// CHECK-NEXT:    %12 = "neura.add"(%9, %0) : (index, index) -> index
// CHECK-NEXT:    %13 = builtin.unrealized_conversion_cast %12 : index to i64
// CHECK-NEXT:    llvm.br ^bb3(%13 : i64)
// CHECK-NEXT:  ^bb5:  // pred: ^bb3
// CHECK-NEXT:    %14 = "neura.add"(%5, %0) : (index, index) -> index
// CHECK-NEXT:    %15 = builtin.unrealized_conversion_cast %14 : index to i64
// CHECK-NEXT:    llvm.br ^bb1(%15 : i64)
// CHECK-NEXT:  ^bb6:  // pred: ^bb1
// CHECK-NEXT:    return
// CHECK-NEXT:  }