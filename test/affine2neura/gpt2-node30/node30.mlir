// Check that the affine loop nest is correctly transformed to neura.loop_control
// RUN: mlir-neura-opt %s --assign-accelerator --lower-affine-to-neura | FileCheck %s
module attributes {} {
  memref.global @A : memref<1x4x16x64xf32> = uninitialized
  memref.global @C : memref<1x4x16x64xf32> = uninitialized
  func.func @_Z6node30v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+01 : f32
    %0 = llvm.mlir.undef : i32
    %1 = memref.get_global @C : memref<1x4x16x64xf32>
    %2 = memref.get_global @A : memref<1x4x16x64xf32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 16 {
        affine.for %arg2 = 0 to 64 {
          %3 = affine.load %2[0, %arg0, %arg1, %arg2] : memref<1x4x16x64xf32>
          %4 = arith.mulf %3, %cst : f32
          affine.store %4, %1[0, %arg0, %arg1, %arg2] : memref<1x4x16x64xf32>
        }
      }
    }
    return %0 : i32
  }
}

// Verify function signature is preserved
// CHECK-LABEL: func.func @_Z6node30v() -> i32

// Verify all affine operations are eliminated
// CHECK-NOT: affine.for
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
// CHECK-NOT: affine.apply

// CHECK-COUNT-3: neura.loop_control
