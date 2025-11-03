// RUN: not mlir-neura-opt %s --lower-affine-to-neura 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// Unsupported Case: affine.if conditional
// This test verifies that direct lowering to Neura fails with a clear error
// when encountering unsupported affine.if operations
module {
  func.func @affine_if_example(%arg0: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
        %val = affine.load %arg0[%i] : memref<10xf32>
      }
    }
    return
  }
}

// ============================================================================
// Test that direct lowering to Neura fails with clear error
// ============================================================================
// CHECK-ERROR: error:
// CHECK-ERROR: affine.if
