// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task --optimize-task-graph \
// RUN: | FileCheck %s

// Tests hyperblock fusion with independent loops that have different operations.

module {
  func.func @test_fusion_with_outputs(%A: memref<16xf32>, %B: memref<16xf32>) {
    // First loop: writes to A.
    affine.for %i = 0 to 16 {
      %idx = arith.index_cast %i : index to i32
      %val = arith.sitofp %idx : i32 to f32
      affine.store %val, %A[%i] : memref<16xf32>
    }
    
    // Second loop: writes to B - independent from first loop.
    affine.for %i = 0 to 16 {
      %idx = arith.index_cast %i : index to i32
      %val = arith.sitofp %idx : i32 to f32
      %doubled = arith.mulf %val, %val : f32
      affine.store %doubled, %B[%i] : memref<16xf32>
    }
    
    return
  }
}

// After conversion and optimization, both loops become separate tasks.

// CHECK:      module {
// CHECK-NEXT:   func.func @test_fusion_with_outputs(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
// CHECK-NEXT:     %memory_outputs = "taskflow.task"(%arg0) <{operandSegmentSizes = array<i32: 1, 0>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_0"}> ({
// CHECK-NEXT:     ^bb0(%arg2: memref<16xf32>):
// CHECK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 16 : index} : index
// CHECK-NEXT:       taskflow.hyperblock indices(%0 : index) {
// CHECK-NEXT:       ^bb0(%arg3: index):
// CHECK-NEXT:         %1 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:         %2 = arith.sitofp %1 : i32 to f32
// CHECK-NEXT:         memref.store %2, %arg2[%arg3] : memref<16xf32>
// CHECK-NEXT:       } -> ()
// CHECK-NEXT:       "taskflow.yield"(%arg2) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<16xf32>) -> ()
// CHECK-NEXT:     }) : (memref<16xf32>) -> memref<16xf32>
// CHECK-NEXT:     %memory_outputs_0 = "taskflow.task"(%arg1) <{operandSegmentSizes = array<i32: 1, 0>, resultSegmentSizes = array<i32: 1, 0>, task_name = "Task_1"}> ({
// CHECK-NEXT:     ^bb0(%arg2: memref<16xf32>):
// CHECK-NEXT:       %0 = taskflow.counter attributes {lower_bound = 0 : index, step = 1 : index, upper_bound = 16 : index} : index
// CHECK-NEXT:       taskflow.hyperblock indices(%0 : index) {
// CHECK-NEXT:       ^bb0(%arg3: index):
// CHECK-NEXT:         %1 = arith.index_cast %arg3 : index to i32
// CHECK-NEXT:         %2 = arith.sitofp %1 : i32 to f32
// CHECK-NEXT:         %3 = arith.mulf %2, %2 : f32
// CHECK-NEXT:         memref.store %3, %arg2[%arg3] : memref<16xf32>
// CHECK-NEXT:       } -> ()
// CHECK-NEXT:       "taskflow.yield"(%arg2) <{operandSegmentSizes = array<i32: 1, 0>}> : (memref<16xf32>) -> ()
// CHECK-NEXT:     }) : (memref<16xf32>) -> memref<16xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
