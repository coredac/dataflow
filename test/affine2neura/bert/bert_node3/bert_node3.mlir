// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-llvm-to-neura | FileCheck %s
module attributes {} {
  func.func @_Z10bert_node3PA128_A768_KfS2_PA128_A768_f(%arg0: memref<?x128x768xf32>, %arg1: memref<?x128x768xf32>, %arg2: memref<?x128x768xf32>) attributes {} {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 768 {
        %0 = affine.load %arg0[0, %arg3, %arg4] : memref<?x128x768xf32>
        %1 = affine.load %arg1[0, %arg3, %arg4] : memref<?x128x768xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %arg2[0, %arg3, %arg4] : memref<?x128x768xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @_Z10bert_node3PA128_A768_KfS2_PA128_A768_f
// CHECK-NOT: arith.
// CHECK-NOT: affine.
// CHECK-NOT: llvm.
