// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Test case for complex affine expressions that need affine.apply
// As suggested by reviewer: when we cannot directly lower affine->neura,
// we emit affine.apply which can later be lowered via affine->scf->neura

module {
  func.func @complex_affine_expr(%arg0: memref<100x100xi32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 10 {
        // Simple case: d0 + cst can be directly lowered
        %idx = affine.apply affine_map<(d0) -> (d0 + 5)>(%i)
        %v = affine.load %arg0[%idx, %j] : memref<100x100xi32>
        affine.store %v, %arg0[%i, %j] : memref<100x100xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @complex_affine_expr
// CHECK: %[[GRANT1:.*]] = neura.grant_once
// CHECK: %[[I:.*]], %[[VALID1:.*]] = neura.loop_control
// CHECK-SAME: <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}>
// CHECK: %[[GRANT2:.*]] = neura.grant_once
// CHECK: %[[J:.*]], %[[VALID2:.*]] = neura.loop_control
// CHECK-SAME: <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}>
// CHECK: %[[CST:.*]] = neura.constant
// CHECK: %[[IDX:.*]] = neura.add %[[I]], %[[CST]]
// CHECK: neura.load_indexed %arg0[%[[IDX]], %[[J]]
// CHECK: neura.store_indexed
// CHECK-NOT: affine.apply
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
