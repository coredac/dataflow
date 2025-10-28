// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Test 1: Perfect nested loops - should reuse valid signals
// CHECK-LABEL: func.func @perfect_nest_2d
func.func @perfect_nest_2d(%A: memref<10x20xf32>) {
  // CHECK: [[GRANT:%.*]] = neura.grant_once
  // CHECK: [[I:%.*]], [[VALID_OUTER:%.*]] = neura.loop_control([[GRANT]])
  // CHECK-SAME: start = 0{{.*}}end = 10
  
  // CHECK-NOT: neura.grant_once
  // CHECK: [[J:%.*]], [[VALID_INNER:%.*]] = neura.loop_control([[VALID_OUTER]])
  // CHECK-SAME: start = 0{{.*}}end = 20
  
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      %v = affine.load %A[%i, %j] : memref<10x20xf32>
    }
  }
  return
}

// Test 2: Triple nested loops - should reuse valid signals transitively
// CHECK-LABEL: func.func @perfect_nest_3d
func.func @perfect_nest_3d(%A: memref<10x20x30xf32>) {
  // CHECK: [[GRANT:%.*]] = neura.grant_once
  // CHECK: [[I:%.*]], [[V1:%.*]] = neura.loop_control([[GRANT]])
  // CHECK-SAME: start = 0{{.*}}end = 10
  
  // CHECK-NOT: neura.grant_once
  // CHECK: [[J:%.*]], [[V2:%.*]] = neura.loop_control([[V1]])
  // CHECK-SAME: start = 0{{.*}}end = 20
  
  // CHECK-NOT: neura.grant_once
  // CHECK: [[K:%.*]], [[V3:%.*]] = neura.loop_control([[V2]])
  // CHECK-SAME: start = 0{{.*}}end = 30
  
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
func.func @two_top_level_loops(%A: memref<10xf32>, %B: memref<20xf32>) {
  // CHECK: [[GRANT1:%.*]] = neura.grant_once
  // CHECK: [[I:%.*]], {{.*}} = neura.loop_control([[GRANT1]])
  affine.for %i = 0 to 10 {
    %v = affine.load %A[%i] : memref<10xf32>
  }
  
  // CHECK: [[GRANT2:%.*]] = neura.grant_once
  // CHECK: [[J:%.*]], {{.*}} = neura.loop_control([[GRANT2]])
  affine.for %j = 0 to 20 {
    %w = affine.load %B[%j] : memref<20xf32>
  }
  return
}

// Test 5: Siblings - two inner loops should both reuse parent's valid
// CHECK-LABEL: func.func @sibling_loops
func.func @sibling_loops(%A: memref<10x20xf32>, %B: memref<10x20xf32>) {
  // CHECK: [[GRANT:%.*]] = neura.grant_once
  // CHECK: [[I:%.*]], [[VALID_OUTER:%.*]] = neura.loop_control([[GRANT]])
  
  affine.for %i = 0 to 10 {
    // First inner loop
    // CHECK-NOT: neura.grant_once
    // CHECK: [[J1:%.*]], {{.*}} = neura.loop_control([[VALID_OUTER]])
    affine.for %j = 0 to 20 {
      %v = affine.load %A[%i, %j] : memref<10x20xf32>
    }
    
    // Second inner loop (sibling)
    // CHECK-NOT: neura.grant_once
    // CHECK: [[J2:%.*]], {{.*}} = neura.loop_control([[VALID_OUTER]])
    affine.for %k = 0 to 20 {
      %w = affine.load %B[%i, %k] : memref<10x20xf32>
    }
  }
  return
}
