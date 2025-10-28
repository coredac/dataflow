// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Stress test 4: Nested loops with constant indices (edge case)
module {
  func.func @constant_indices(%arg0: memref<10x10xi32>) {
    affine.for %i = 0 to 5 {
      affine.for %j = 0 to 5 {
        // Load from constant index
        %v = affine.load %arg0[0, 0] : memref<10x10xi32>
        // Store using loop indices
        affine.store %v, %arg0[%i, %j] : memref<10x10xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @constant_indices
// CHECK: %[[GRANT1:.*]] = "neura.grant_once"
// CHECK: %[[I:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT1]])
// CHECK: %[[GRANT2:.*]] = "neura.grant_once"
// CHECK: %[[J:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT2]])
// Load with constant indices
// CHECK: %[[C0_1:.*]] = "neura.constant"() <{value = 0 : index}>
// CHECK: %[[C0_2:.*]] = "neura.constant"() <{value = 0 : index}>
// CHECK: neura.load_indexed %arg0[%[[C0_1]], %[[C0_2]]
// Store with loop indices
// CHECK: neura.store_indexed %{{.*}} to %arg0[%[[I]], %[[J]]
