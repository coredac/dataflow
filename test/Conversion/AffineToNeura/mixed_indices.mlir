// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Stress test 5: Mix of direct indices and affine expressions
module {
  func.func @mixed_indices(%arg0: memref<100x100xi32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 10 {
        // Use affine.apply for index calculation: i+1, j+2
        %idx_i = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
        %idx_j = affine.apply affine_map<(d0) -> (d0 + 2)>(%j)
        %v = affine.load %arg0[%idx_i, %idx_j] : memref<100x100xi32>
        affine.store %v, %arg0[%i, %j] : memref<100x100xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @mixed_indices
// CHECK: %[[GRANT1:.*]] = "neura.grant_once"
// CHECK: %[[I:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT1]])
// CHECK: %[[GRANT2:.*]] = "neura.grant_once"
// CHECK: %[[J:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT2]])
// Check affine.apply is converted to neura.add
// CHECK: %[[C1:.*]] = "neura.constant"() <{value = 1 : index}>
// CHECK: %[[IDX_I:.*]] = neura.add %[[I]], %[[C1]]
// CHECK: %[[C2:.*]] = "neura.constant"() <{value = 2 : index}>
// CHECK: %[[IDX_J:.*]] = neura.add %[[J]], %[[C2]]
// CHECK: neura.load_indexed %arg0[%[[IDX_I]], %[[IDX_J]]
// CHECK: neura.store_indexed %{{.*}} to %arg0[%[[I]], %[[J]]
// CHECK-NOT: affine.apply
