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

// CHECK-LABEL: func.func @deep_nesting_4d(%arg0: memref<5x5x5x5xf32>)
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 5 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : index, index, index, index] memref<5x5x5x5xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NOT: affine.
