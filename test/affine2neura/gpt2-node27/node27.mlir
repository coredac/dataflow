// Check that the affine loop nest is correctly transformed to neura.loop_control
// RUN: mlir-neura-opt %s --assign-accelerator --lower-affine-to-neura | FileCheck %s
module attributes {} {
  memref.global @input : memref<1x16x4x16xf32> = uninitialized
  memref.global @output : memref<1x4x16x16xf32> = uninitialized
  func.func @_Z6node27v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.undef : i32
    %1 = memref.get_global @output : memref<1x4x16x16xf32>
    %2 = memref.get_global @input : memref<1x16x4x16xf32>
    affine.for %arg0 = 0 to 16 {
      affine.for %arg1 = 0 to 4 {
        affine.for %arg2 = 0 to 16 {
          %3 = affine.load %2[0, %arg1, %arg0, %arg2] : memref<1x16x4x16xf32>
          affine.store %3, %1[0, %arg0, %arg1, %arg2] : memref<1x4x16x16xf32>
        }
      }
    }
    return %0 : i32
  }
}
// Verify function signature is preserved
// CHECK-LABEL: func.func @_Z6node27v() -> i32

// Verify all affine operations are eliminated
// CHECK-NOT: affine.for
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
// CHECK-NOT: affine.apply

// CHECK-COUNT-3: neura.loop_control
