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

// CHECK-LABEL: func.func @imperfect_ops_after(%arg0: memref<10x20xf32>, %arg1: memref<10xf32>)
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}}, %{{.*}} : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: %{{.*}} = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: neura.store_indexed %{{.*}} to %arg1[%{{.*}} : index] memref<10xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NOT: affine.
