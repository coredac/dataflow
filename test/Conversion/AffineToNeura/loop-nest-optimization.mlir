// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// This test verifies proper handling of various loop nest patterns.

module {
  func.func @perfect_nest_2d(%arg0: memref<10x20xf32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        %v = affine.load %arg0[%i, %j] : memref<10x20xf32>
      }
    }
    return
  }

  func.func @two_top_level_loops(%arg0: memref<10xf32>, %arg1: memref<20xf32>) {
    affine.for %i = 0 to 10 {
      %v = affine.load %arg0[%i] : memref<10xf32>
    }
    affine.for %j = 0 to 20 {
      %w = affine.load %arg1[%j] : memref<20xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @perfect_nest_2d(%arg0: memref<10x20xf32>)
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}}, %{{.*}} : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @two_top_level_loops(%arg0: memref<10xf32>, %arg1: memref<20xf32>)
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}} : index] memref<10xf32> : f32
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg1[%{{.*}} : index] memref<20xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }
