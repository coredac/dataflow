// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-llvm-to-neura | FileCheck %s
module attributes {} {
  func.func @_Z10bert_node9PA128_A768_KfPA128_A768_d(%arg0: memref<?x128x768xf32>, %arg1: memref<?x128x768xf64>) attributes {} {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %0 = affine.load %arg0[0, %arg2, %arg3] : memref<?x128x768xf32>
        %1 = arith.extf %0 : f32 to f64
        affine.store %1, %arg1[0, %arg2, %arg3] : memref<?x128x768xf64>
      }
    }
    return
  }
}


// CHECK-LABEL: func.func @_Z10bert_node9PA128_A768_KfPA128_A768_d
// CHECK-NOT: arith.
// CHECK-NOT: affine.
// CHECK-NOT: llvm.