// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Stress test 1: Triple nested loops with multiple memory accesses
module {
  func.func @triple_nested_loop(%arg0: memref<64x64x64xi32>, %arg1: memref<64x64x64xi32>) {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.for %k = 0 to 8 {
          %v1 = affine.load %arg0[%i, %j, %k] : memref<64x64x64xi32>
          %v2 = affine.load %arg1[%i, %j, %k] : memref<64x64x64xi32>
          affine.store %v1, %arg1[%i, %j, %k] : memref<64x64x64xi32>
          affine.store %v2, %arg0[%i, %j, %k] : memref<64x64x64xi32>
        }
      }
    }
    return
  }
}

// Verify that we have three grant_once and three loop_control operations
// CHECK-LABEL: func.func @triple_nested_loop
// CHECK: %[[GRANT1:.*]] = "neura.grant_once"
// CHECK: %[[I:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT1]])
// CHECK-SAME: end = 8
// CHECK: %[[GRANT2:.*]] = "neura.grant_once"
// CHECK: %[[J:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT2]])
// CHECK-SAME: end = 8
// CHECK: %[[GRANT3:.*]] = "neura.grant_once"
// CHECK: %[[K:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT3]])
// CHECK-SAME: end = 8
// CHECK: neura.load_indexed %arg0[%[[I]], %[[J]], %[[K]]
// CHECK: neura.load_indexed %arg1[%[[I]], %[[J]], %[[K]]
// CHECK: neura.store_indexed %{{.*}} to %arg1[%[[I]], %[[J]], %[[K]]
// CHECK: neura.store_indexed %{{.*}} to %arg0[%[[I]], %[[J]], %[[K]]
// CHECK-NOT: affine.for
