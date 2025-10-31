// RUN: not mlir-neura-opt %s --lower-affine-to-neura 2>&1 | FileCheck %s

// Unsupported Case: affine.if conditional
// This test demonstrates what happens when lowering encounters unsupported operations
module {
  func.func @affine_if_example(%arg0: memref<10xf32>, %N: index) {
    affine.for %i = 0 to 10 {
      affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
        %val = affine.load %arg0[%i] : memref<10xf32>
      }
    }
    return
  }
}

// ============================================================================
// What happens when lowering fails:
// ============================================================================
// 1. Pass encounters affine.if operation (not in conversion target)
// 2. Error is emitted indicating failed legalization
// 3. Affine operations remain unchanged in the IR
//
// CHECK: error:
// CHECK: affine.if
//
// Note: affine.if is not currently supported in this direct lowering pass.
// Alternative lowering path:
//   1. Use --lower-affine-to-loops to convert affine.if -> scf.if
//   2. Use --convert-scf-to-cf to convert scf.if -> cf.cond_br
//   3. Then use a separate pass to convert control flow to Neura predicated ops
// This multi-stage approach provides more flexibility for handling conditionals.
// ============================================================================
