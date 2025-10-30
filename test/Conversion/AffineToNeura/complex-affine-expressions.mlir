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
  // CHECK-LABEL: func.func @mul_expression
  // CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
  // CHECK-NEXT: %[[I:.*]], %[[VALID:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
  // CHECK-NEXT: %[[C2:.*]] = "neura.constant"() <{value = 2 : index}> : () -> index
  // CHECK-NEXT: %[[MUL:.*]] = "neura.mul"(%[[I]], %[[C2]]) : (index, index) -> index
  // CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[MUL]] : index] memref<10xf32> : f32
  // CHECK-NEXT: return

  // Test 2: Addition and multiplication (d0 * 2 + 1)
  func.func @complex_expression(%arg0: memref<100xf32>) {
    affine.for %i = 0 to 10 {
      %0 = affine.load %arg0[2 * %i + 1] : memref<100xf32>
    }
    return
  }
  // CHECK-LABEL: func.func @complex_expression
  // CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
  // CHECK-NEXT: %[[I:.*]], %[[VALID:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
  // CHECK-NEXT: %[[C2:.*]] = "neura.constant"() <{value = 2 : index}> : () -> index
  // CHECK-NEXT: %[[MUL:.*]] = "neura.mul"(%[[I]], %[[C2]]) : (index, index) -> index
  // CHECK-NEXT: %[[C1:.*]] = "neura.constant"() <{value = 1 : index}> : () -> index
  // CHECK-NEXT: %[[ADD:.*]] = "neura.add"(%[[MUL]], %[[C1]]) : (index, index) -> index
  // CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[ADD]] : index] memref<100xf32> : f32
  // CHECK-NEXT: return

  // Test 3: Modulo operation (d0 % 8)
  func.func @modulo_expression(%arg0: memref<64xf32>) {
    affine.for %i = 0 to 64 {
      %0 = affine.load %arg0[%i mod 8] : memref<64xf32>
    }
    return
  }
  // CHECK-LABEL: func.func @modulo_expression
  // CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
  // CHECK-NEXT: %[[I:.*]], %[[VALID:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 64 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
  // CHECK-NEXT: %[[C8:.*]] = "neura.constant"() <{value = 8 : index}> : () -> index
  // CHECK-NEXT: %[[REM:.*]] = "neura.rem"(%[[I]], %[[C8]]) : (index, index) -> index
  // CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[REM]] : index] memref<64xf32> : f32
  // CHECK-NEXT: return

  // Test 4: Floor division and modulo with affine.apply
  // Note: affine.apply operations are expanded into explicit arithmetic ops
  func.func @floordiv_expression(%arg0: memref<8x8xf32>) {
    affine.for %i = 0 to 32 {
      %row = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%i)
      %col = affine.apply affine_map<(d0) -> (d0 mod 4)>(%i)
      %0 = affine.load %arg0[%row, %col] : memref<8x8xf32>
    }
    return
  }
  // CHECK-LABEL: func.func @floordiv_expression
  // CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
  // CHECK-NEXT: %[[I:.*]], %[[VALID:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 32 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
  // CHECK-NEXT: %[[C4_1:.*]] = "neura.constant"() <{value = 4 : index}> : () -> index
  // CHECK-NEXT: %[[DIV:.*]] = "neura.div"(%[[I]], %[[C4_1]]) : (index, index) -> index
  // CHECK-NEXT: %[[C4_2:.*]] = "neura.constant"() <{value = 4 : index}> : () -> index
  // CHECK-NEXT: %[[REM:.*]] = "neura.rem"(%[[I]], %[[C4_2]]) : (index, index) -> index
  // CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[DIV]], %[[REM]] : index, index] memref<8x8xf32> : f32
  // CHECK-NEXT: return

  // Test 5: Multiple dimensions with complex expressions (max 2D for CGRA support)
  func.func @multi_dim_complex(%arg0: memref<10x20xf32>) {
    affine.for %i = 0 to 10 {
      affine.for %j = 0 to 20 {
        %0 = affine.load %arg0[%i, %j + 1] : memref<10x20xf32>
      }
    }
    return
  }
  // CHECK-LABEL: func.func @multi_dim_complex
  // CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
  // CHECK-NEXT: %[[I:.*]], %[[VALID_I:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
  // CHECK-NEXT: %[[J:.*]], %[[VALID_J:.*]] = "neura.loop_control"(%[[VALID_I]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
  // CHECK-NEXT: %[[C1:.*]] = "neura.constant"() <{value = 1 : index}> : () -> index
  // CHECK-NEXT: %[[ADD:.*]] = "neura.add"(%[[J]], %[[C1]]) : (index, index) -> index
  // CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[ADD]] : index, index] memref<10x20xf32> : f32
  // CHECK-NEXT: return
}
