// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Stress test 2: Loop with non-zero lower bound and custom step
module {
  func.func @custom_bounds(%arg0: memref<100xi32>) {
    affine.for %i = 5 to 50 step 3 {
      %v = affine.load %arg0[%i] : memref<100xi32>
      affine.store %v, %arg0[%i] : memref<100xi32>
    }
    return
  }
}

// CHECK-LABEL: func.func @custom_bounds
// CHECK: %[[GRANT:.*]] = "neura.grant_once"
// CHECK: %[[IDX:.*]], %{{.*}} = "neura.loop_control"(%[[GRANT]])
// CHECK-SAME: <{end = 50 : i64, iterationType = "increment", start = 5 : i64, step = 3 : i64}>
// CHECK: neura.load_indexed %arg0[%[[IDX]]
// CHECK: neura.store_indexed %{{.*}} to %arg0[%[[IDX]]
