// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-llvm-to-neura | FileCheck %s
module attributes {} {
  func.func @_Z10bert_node8PA128_A1_KfPA128_A1_f(%arg0: memref<?x128x1xf32>, %arg1: memref<?x128x1xf32>) attributes {} {
    %cst = arith.constant 7.680000e+02 : f32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[0, %arg2, 0] : memref<?x128x1xf32>
      %1 = arith.divf %0, %cst : f32
      affine.store %1, %arg1[0, %arg2, 0] : memref<?x128x1xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @_Z10bert_node8PA128_A1_KfPA128_A1_f
// CHECK-NOT: arith.
// CHECK-NOT: affine.
// CHECK-NOT: llvm.
