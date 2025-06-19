// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-llvm-to-neura | FileCheck %s

module attributes {} {
  func.func @_Z10bert_node0PA128_KiPA128_b(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi8>) attributes {} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[0, %arg2] : memref<?x128xi32>
      %1 = arith.cmpi sgt, %0, %c0_i32 : i32
      %2 = arith.extui %1 : i1 to i8
      affine.store %2, %arg1[0, %arg2] : memref<?x128xi8>
    }
    return
  }
}

// CHECK-LABEL: func.func @_Z10bert_node0PA128_KiPA128_b
// CHECK-NOT: arith.
// CHECK-NOT: affine.
// CHECK-NOT: llvm.

