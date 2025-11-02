// RUN: not mlir-neura-opt %s --lower-affine-to-neura 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: mlir-neura-opt %s --lower-affine | FileCheck %s --check-prefix=CHECK-SCF

// Unsupported Case: affine.if conditional
// This test demonstrates:
// 1. Direct lowering to Neura fails (affine.if not supported)
// 2. Alternative multi-stage lowering path via SCF dialect
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
// CHECK-ERROR: Test that direct lowering to Neura fails with clear error
// ============================================================================
// CHECK-ERROR: error:
// CHECK-ERROR: affine.if

// ============================================================================
// CHECK-SCF: Alternative lowering path: affine -> scf
// This demonstrates the first stage of multi-stage lowering:
//   1. affine.if -> scf.if (shown here)
//   2. scf.if -> cf.cond_br (would use --convert-scf-to-cf)
//   3. cf ops -> neura predicated ops (requires separate pass)
// ============================================================================
// CHECK-SCF-LABEL: func.func @affine_if_example
// CHECK-SCF: scf.for
// CHECK-SCF: scf.if
// CHECK-SCF: memref.load
