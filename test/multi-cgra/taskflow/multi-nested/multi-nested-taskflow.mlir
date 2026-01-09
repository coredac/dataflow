// RUN: mlir-neura-opt %s | FileCheck %s

module {
  func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_(%arg0: memref<?x8x6xi32>, %arg1: memref<?x8x5xi32>, %arg2: memref<?x8x5xi32>, %arg3: memref<?x7xi32>, %arg4: memref<?x9xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>, %arg8: memref<?xi32>, %arg9: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    taskflow.graph(%arg0, %arg5, %arg1, %arg2, %arg6, %arg9, %arg3, %arg7, %arg4, %arg8) {
    ^bb0(%arg10: memref<?x8x6xi32>, %arg11: memref<?xi32>, %arg12: memref<?x8x5xi32>, %arg13: memref<?x8x5xi32>, %arg14: memref<?xi32>, %arg15: memref<?xi32>, %arg16: memref<?x7xi32>, %arg17: memref<?xi32>, %arg18: memref<?x9xi32>, %arg19: memref<?xi32>):
      %data_outs = "taskflow.task"(%arg10, %arg11) <{operandSegmentSizes = array<i32: 0, 2>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
      ^bb0(%arg20: memref<?x8x6xi32>, %arg21: memref<?xi32>):
        affine.for %arg22 = 0 to 4 {
          affine.for %arg23 = 0 to 8 {
            affine.for %arg24 = 0 to 6 {
              %4 = affine.load %arg20[%arg22, %arg23, %arg24] : memref<?x8x6xi32>
              affine.store %4, %arg21[%arg24] : memref<?xi32>
            }
          }
        }
        taskflow.yield %arg21 : memref<?xi32>
      }) : (memref<?x8x6xi32>, memref<?xi32>) -> memref<?xi32>
      %1 = taskflow.channel %data_outs : memref<?xi32> -> memref<?xi32>
      %data_outs_0 = "taskflow.task"(%arg12, %arg13, %arg14) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_1"}> ({
      ^bb0(%arg20: memref<?x8x5xi32>, %arg21: memref<?x8x5xi32>, %arg22: memref<?xi32>):
        affine.for %arg23 = 0 to 4 {
          affine.for %arg24 = 0 to 8 {
            affine.for %arg25 = 0 to 5 {
              %4 = affine.load %arg20[%arg23, %arg24, %arg25] : memref<?x8x5xi32>
              %5 = affine.load %arg21[%arg23, %arg24, %arg25] : memref<?x8x5xi32>
              %6 = arith.addi %4, %5 : i32
              affine.store %6, %arg22[%arg25] : memref<?xi32>
            }
          }
        }
        taskflow.yield %arg22 : memref<?xi32>
      }) : (memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>) -> memref<?xi32>
      %2 = taskflow.channel %data_outs_0 : memref<?xi32> -> memref<?xi32>
      %data_outs_1 = "taskflow.task"(%1, %2, %arg15) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_2"}> ({
      ^bb0(%arg20: memref<?xi32>, %arg21: memref<?xi32>, %arg22: memref<?xi32>):
        affine.for %arg23 = 0 to 4 {
          affine.for %arg24 = 0 to 8 {
            affine.for %arg25 = 0 to 6 {
              %4 = affine.load %arg20[%arg25] : memref<?xi32>
              %5 = affine.load %arg21[%arg25] : memref<?xi32>
              %6 = arith.addi %4, %5 : i32
              %7 = affine.load %arg22[0] : memref<?xi32>
              %8 = arith.addi %7, %6 : i32
              affine.store %8, %arg22[0] : memref<?xi32>
            }
          }
        }
        taskflow.yield %arg22 : memref<?xi32>
      }) : (memref<?xi32>, memref<?xi32>, memref<?xi32>) -> memref<?xi32>
      %data_outs_2 = "taskflow.task"(%arg16, %arg17) <{operandSegmentSizes = array<i32: 0, 2>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_3"}> ({
      ^bb0(%arg20: memref<?x7xi32>, %arg21: memref<?xi32>):
        affine.for %arg22 = 0 to 4 {
          affine.for %arg23 = 0 to 7 {
            %4 = affine.load %arg20[%arg22, %arg23] : memref<?x7xi32>
            affine.store %4, %arg21[%arg23] : memref<?xi32>
          }
        }
        taskflow.yield %arg21 : memref<?xi32>
      }) : (memref<?x7xi32>, memref<?xi32>) -> memref<?xi32>
      %3 = taskflow.channel %data_outs_2 : memref<?xi32> -> memref<?xi32>
      %data_outs_3 = "taskflow.task"(%arg18, %3, %arg19) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_4"}> ({
      ^bb0(%arg20: memref<?x9xi32>, %arg21: memref<?xi32>, %arg22: memref<?xi32>):
        affine.for %arg23 = 0 to 4 {
          affine.for %arg24 = 0 to 9 {
            %4 = affine.load %arg20[%arg23, %arg24] : memref<?x9xi32>
            %5 = affine.load %arg21[%arg24] : memref<?xi32>
            %6 = arith.addi %4, %5 : i32
            affine.store %6, %arg22[%arg24] : memref<?xi32>
          }
        }
        taskflow.yield %arg22 : memref<?xi32>
      }) : (memref<?x9xi32>, memref<?xi32>, memref<?xi32>) -> memref<?xi32>
    } : (memref<?x8x6xi32>, memref<?xi32>, memref<?x8x5xi32>, memref<?x8x5xi32>, memref<?xi32>, memref<?xi32>, memref<?x7xi32>, memref<?xi32>, memref<?x9xi32>, memref<?xi32>) -> ()
    %0 = affine.load %arg9[0] : memref<?xi32>
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @_Z21pureNestedLoopExamplePA8_A6_iPA8_A5_iS4_PA7_iPA9_iPiS9_S9_S9_S9_
// CHECK: taskflow.graph
// CHECK: taskflow.task
// CHECK: taskflow.channel
// CHECK: taskflow.yield