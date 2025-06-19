// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-llvm-to-neura | FileCheck %s
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

// CHECK-LABEL: func.func @_Z10bert_node1PA1_A1_A1_A1_A128_bPA1_A128_S1_
// CHECK-NOT: arith.
// CHECK-NOT: affine.
// CHECK-NOT: llvm.
