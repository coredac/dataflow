// Check that the affine loop nest is correctly transformed to neura.loop_control
// RUN: mlir-neura-opt %s --assign-accelerator --lower-affine-to-neura | FileCheck %s
module attributes {} {
  memref.global @input : memref<1x16x64xf32> = uninitialized
  memref.global @output : memref<1x16xf32> = uninitialized
  func.func @_Z6node11v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @output : memref<1x16xf32>
    %1 = memref.get_global @input : memref<1x16x64xf32>
    affine.for %arg0 = 0 to 16 {
      affine.for %arg1 = 0 to 64 {
        %2 = affine.load %1[0, %arg0, %arg1] : memref<1x16x64xf32>
        %3 = affine.load %0[0, %arg0] : memref<1x16xf32>
        %4 = arith.addf %3, %2 : f32
        affine.store %4, %0[0, %arg0] : memref<1x16xf32>
      }
    }
    return %c0_i32 : i32
  }
}

// Verify function signature is preserved
// CHECK-LABEL: func.func @_Z6node11v() -> i32

// Verify all affine operations are eliminated
// CHECK-NOT: affine.for
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
// CHECK-NOT: affine.apply

// CHECK-COUNT-2: neura.loop_control
