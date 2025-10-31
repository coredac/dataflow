// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Imperfect Nesting: Operations after child loop
// This tests that inner loop results can be used by outer loop operations
module {
  func.func @imperfect_ops_after(%arg0: memref<10x20xf32>, %arg1: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      // Inner loop: compute sum of row elements
      affine.for %j = 0 to 20 {
        %elem = affine.load %arg0[%i, %j] : memref<10x20xf32>
        // In real code, %elem would be accumulated or used
      }
      // Operations after inner loop - uses outer loop index
      %cst = arith.constant 1.0 : f32
      affine.store %cst, %arg1[%i] : memref<10xf32>
    }
    return
  }
}

// ============================================================================
// Verify transformation: no affine ops, valid signal reuse for inner loop
// ============================================================================
// CHECK-LABEL: func.func @imperfect_ops_after(%arg0: memref<10x20xf32>, %arg1: memref<10xf32>)
// CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VI:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: //
// CHECK-NEXT: %[[J:.*]], %[[VJ:.*]] = "neura.loop_control"(%[[VI]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[J]] : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: //
// CHECK-NEXT: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: neura.store_indexed %[[CST]] to %arg1[%[[I]] : index] memref<10xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NOT: affine.
