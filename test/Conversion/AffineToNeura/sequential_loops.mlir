// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Stress test 3: Multiple sequential loops (not nested)
module {
  func.func @sequential_loops(%arg0: memref<100xi32>, %arg1: memref<100xi32>) {
    affine.for %i = 0 to 10 {
      %v = affine.load %arg0[%i] : memref<100xi32>
      affine.store %v, %arg1[%i] : memref<100xi32>
    }
    affine.for %j = 0 to 20 {
      %v = affine.load %arg1[%j] : memref<100xi32>
      affine.store %v, %arg0[%j] : memref<100xi32>
    }
    return
  }
}

// CHECK-LABEL: func.func @sequential_loops
// First loop
// CHECK: %[[GRANT1:.*]] = "neura.grant_once"
// CHECK: %[[I:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT1]])
// CHECK-SAME: end = 10
// CHECK: neura.load_indexed %arg0[%[[I]]
// CHECK: neura.store_indexed %{{.*}} to %arg1[%[[I]]
// Second loop
// CHECK: %[[GRANT2:.*]] = "neura.grant_once"
// CHECK: %[[J:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT2]])
// CHECK-SAME: end = 20
// CHECK: neura.load_indexed %arg1[%[[J]]
// CHECK: neura.store_indexed %{{.*}} to %arg0[%[[J]]
