// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Test 1: Perfect nested loops - should reuse valid signals
// CHECK-LABEL: func.func @perfect_nest_2d
// CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VALID_I:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[J:.*]], %[[VALID_J:.*]] = "neura.loop_control"(%[[VALID_I]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[J]] : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: return
func.func @perfect_nest_2d(%A: memref<10x20xf32>) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      %v = affine.load %A[%i, %j] : memref<10x20xf32>
    }
  }
  return
}

// Test 2: Triple nested loops - should reuse valid signals transitively
// CHECK-LABEL: func.func @perfect_nest_3d
// CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VALID_I:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[J:.*]], %[[VALID_J:.*]] = "neura.loop_control"(%[[VALID_I]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[K:.*]], %[[VALID_K:.*]] = "neura.loop_control"(%[[VALID_J]]) <{end = 30 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[J]], %[[K]] : index, index, index] memref<10x20x30xf32> : f32
// CHECK-NEXT: return
func.func @perfect_nest_3d(%A: memref<10x20x30xf32>) {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      affine.for %k = 0 to 30 {
        %v = affine.load %A[%i, %j, %k] : memref<10x20x30xf32>
      }
    }
  }
  return
}

// Test 3: Imperfect nested loop - operations before inner loop
// CHECK-LABEL: func.func @imperfect_nest_before
// CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VALID_I:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[J:.*]], %[[VALID_J:.*]] = "neura.loop_control"(%[[VALID_I]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[J]] : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: return
func.func @imperfect_nest_before(%A: memref<10x20xf32>, %B: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %c = arith.constant 0.0 : f32
    affine.for %j = 0 to 20 {
      %v = affine.load %A[%i, %j] : memref<10x20xf32>
    }
  }
  return
}

// Test 4: Two separate top-level loops - each should get its own grant_once
// CHECK-LABEL: func.func @two_top_level_loops
// CHECK-NEXT: %[[GRANT1:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VALID_I:.*]] = "neura.loop_control"(%[[GRANT1]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]] : index] memref<10xf32> : f32
// CHECK-NEXT: %[[GRANT2:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[J:.*]], %[[VALID_J:.*]] = "neura.loop_control"(%[[GRANT2]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg1[%[[J]] : index] memref<20xf32> : f32
// CHECK-NEXT: return
func.func @two_top_level_loops(%A: memref<10xf32>, %B: memref<20xf32>) {
  affine.for %i = 0 to 10 {
    %v = affine.load %A[%i] : memref<10xf32>
  }
  
  affine.for %j = 0 to 20 {
    %w = affine.load %B[%j] : memref<20xf32>
  }
  return
}

// Test 5: Siblings - two inner loops should both reuse parent's valid
// CHECK-LABEL: func.func @sibling_loops
// CHECK-NEXT: %[[GRANT:.*]] = "neura.grant_once"() : () -> i1
// CHECK-NEXT: %[[I:.*]], %[[VALID_I:.*]] = "neura.loop_control"(%[[GRANT]]) <{end = 10 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %[[J1:.*]], %[[VALID_J1:.*]] = "neura.loop_control"(%[[VALID_I]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[I]], %[[J1]] : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: %[[J2:.*]], %[[VALID_J2:.*]] = "neura.loop_control"(%[[VALID_I]]) <{end = 20 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg1[%[[I]], %[[J2]] : index, index] memref<10x20xf32> : f32
// CHECK-NEXT: return
func.func @sibling_loops(%A: memref<10x20xf32>, %B: memref<10x20xf32>) {
  affine.for %i = 0 to 10 {
    // First inner loop
    affine.for %j = 0 to 20 {
      %v = affine.load %A[%i, %j] : memref<10x20xf32>
    }
    
    // Second inner loop (sibling)
    affine.for %k = 0 to 20 {
      %w = affine.load %B[%i, %k] : memref<10x20xf32>
    }
  }
  return
}
