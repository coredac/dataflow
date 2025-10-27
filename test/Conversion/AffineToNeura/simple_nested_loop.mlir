// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

module {
  func.func @simple_nested_loop(%arg0: memref<?x1x1x1x1x128xi8>, %arg1: memref<?x1x128x1x1x128xi8>) {
    affine.for %i = 0 to 128 {
      affine.for %j = 0 to 128 {
        %0 = affine.load %arg0[0, 0, 0, 0, 0, %j] : memref<?x1x1x1x1x128xi8>
        affine.store %0, %arg1[0, 0, %i, 0, 0, %j] : memref<?x1x128x1x1x128xi8>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @simple_nested_loop
// CHECK: %[[PARENT_VALID:.*]] = neura.grant_once
// CHECK: %[[OUTER_IDX:.*]], %[[OUTER_VALID:.*]] = neura.loop_control
// CHECK-SAME: <{end = 128 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}>
// CHECK: %[[INNER_IDX:.*]], %[[INNER_VALID:.*]] = neura.loop_control
// CHECK-SAME: <{end = 128 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}>
// CHECK: neura.load_indexed
// CHECK: neura.store_indexed
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
