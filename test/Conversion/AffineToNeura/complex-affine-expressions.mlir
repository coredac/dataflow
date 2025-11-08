// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// This test verifies that complex affine expressions are correctly expanded
// into explicit Neura arithmetic operations.

module {
  // Test 1: Multiplication expression (d0 * 2)
  func.func @mul_expression(%arg0: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.load %arg0[2 * %i] : memref<10xf32>
    }
    return
  }

  // Test 2: Addition and multiplication (d0 * 3 + 1)
  func.func @complex_expression(%arg0: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.load %arg0[3 * %i + 1] : memref<10xf32>
    }
    return
  }

  // Test 3: Modulo operation (d0 % 4)
  func.func @modulo_expression(%arg0: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.load %arg0[%i mod 4] : memref<10xf32>
    }
    return
  }

  // Test 4: Floor division (d0 floordiv 2)
  func.func @floordiv_expression(%arg0: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.load %arg0[%i floordiv 2] : memref<10xf32>
    }
    return
  }

  // Test 5: Multiple dimensions with complex expressions
  func.func @multi_dim_complex(%arg0: memref<10x20xf32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        %0 = affine.load %arg0[%i, 2 * %i + 3 * %j + 1] : memref<10x20xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @mul_expression
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
//
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.mul"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}} : index] memref<10xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @complex_expression
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
//
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 3 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.mul"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.add"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}} : index] memref<10xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @modulo_expression
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
//
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.rem"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}} : index] memref<10xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @floordiv_expression
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
//
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.div"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}} : index] memref<10xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @multi_dim_complex
// CHECK-NEXT: %0 = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%0) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}}, %{{.*}} = "neura.loop_control"(%{{.*}}) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
//
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.mul"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 3 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.mul"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = "neura.add"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT: %{{.*}} = "neura.add"(%{{.*}}, %{{.*}}) : (index, index) -> index
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%{{.*}}, %{{.*}} : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }
