// RUN: not mlir-neura-opt %s --lower-affine-to-neura 2>&1 | FileCheck %s
// Note: The "not" command inverts the exit status - expects the pass to fail.
// This allows us to test error handling by checking that the pass correctly
// rejects unsupported input and emits appropriate error messages.

// Unsupported Case: Dynamic loop bounds
// This test demonstrates what happens when lowering fails
module {
  func.func @dynamic_upper_bound(%arg0: memref<?xf32>, %N: index) {
    affine.for %i = 0 to %N {
      %val = affine.load %arg0[%i] : memref<?xf32>
    }
    return
  }
}

// ============================================================================
// What happens when lowering fails:
// ============================================================================
// 1. Pattern matching fails, error is emitted
// 2. Affine operations remain unchanged in the IR
// 3. Pass fails with error message
//
// CHECK: error: [affine2neura] Non-constant loop bounds not supported
// CHECK: affine.for %i = 0 to %N
// CHECK: affine.load
//
// Note: This case is unsupported because neura.loop_control requires
// compile-time constant bounds for CGRA hardware configuration.
// We do not target dynamic bounds in this lowering pass.