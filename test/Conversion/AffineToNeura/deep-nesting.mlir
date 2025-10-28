// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Corner Case: Deeply nested loops (4 levels) - tests perfect nesting with 4D
module {
  func.func @deep_nesting_4d(%arg0: memref<5x5x5x5xf32>) {
    affine.for %i = 0 to 5 {
      affine.for %j = 0 to 5 {
        affine.for %k = 0 to 5 {
          affine.for %l = 0 to 5 {
            %0 = affine.load %arg0[%i, %j, %k, %l] : memref<5x5x5x5xf32>
          }
        }
      }
    }
    return
  }
}

// ============================================================================
// Verify transformation: no affine ops, only neura ops, 1 grant_once for perfect nest
// ============================================================================
// CHECK-LABEL: func.func @deep_nesting_4d
// CHECK-NOT: affine.
// CHECK-NEXT: %[[V0:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VI:.*]] = "neura.loop_control"(%[[V0]]) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[J:.*]], %[[VJ:.*]] = "neura.loop_control"(%[[VI]]) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[K:.*]], %[[VK:.*]] = "neura.loop_control"(%[[VJ]]) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[L:.*]], %[[VL:.*]] = "neura.loop_control"(%[[VK]]) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[J]], %[[K]], %[[L]] : index, index, index, index] memref<5x5x5x5xf32> : f32
// CHECK-NEXT: return
// CHECK-NOT: affine.
