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
// Showing the entire IR to understand what is happening in the pass:
// CHECK-NEXT: %[[GRANT_OUTER:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[OUTER_IDX:.*]], %[[OUTER_VALID:.*]] = "neura.loop_control"(%[[GRANT_OUTER]])
// CHECK-SAME: <{end = 128 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}>
// CHECK-SAME: : (i1) -> (index, i1)
// CHECK-NEXT: %[[GRANT_INNER:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[INNER_IDX:.*]], %[[INNER_VALID:.*]] = "neura.loop_control"(%[[GRANT_INNER]])
// CHECK-SAME: <{end = 128 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}>
// CHECK-SAME: : (i1) -> (index, i1)
// CHECK-NEXT: %[[C0_1:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_2:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_3:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_4:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_5:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[LOADED:.*]] = neura.load_indexed %arg0[%[[C0_1]], %[[C0_2]], %[[C0_3]], %[[C0_4]], %[[C0_5]], %[[INNER_IDX]]
// CHECK-SAME: : index, index, index, index, index, index] memref<?x1x1x1x1x128xi8> : i8
// CHECK-NEXT: %[[C0_6:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_7:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_8:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %[[C0_9:.*]] = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: neura.store_indexed %[[LOADED]] to %arg1[%[[C0_6]], %[[C0_7]], %[[OUTER_IDX]], %[[C0_8]], %[[C0_9]], %[[INNER_IDX]]
// CHECK-SAME: : index, index, index, index, index, index] memref<?x1x128x1x1x128xi8> : i8
// CHECK-NEXT: return
// CHECK-NOT: affine.for
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
