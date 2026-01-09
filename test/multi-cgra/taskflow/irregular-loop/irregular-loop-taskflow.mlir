// RUN: mlir-neura-opt %s | FileCheck %s

module {
  func.func @_Z21irregularLoopExample1v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %alloca_0 = memref.alloca() : memref<4x8xi32>
    taskflow.graph(%c0_i32, %alloca_0, %alloca, %c2_i32, %c8_i32) {
    ^bb0(%arg0: i32, %arg1: memref<4x8xi32>, %arg2: memref<i32>, %arg3: i32, %arg4: i32):
      %data_outs = "taskflow.task"(%arg0) <{operandSegmentSizes = array<i32: 0, 1>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_0"}> ({
      ^bb0(%arg5: i32):
        %7 = affine.for %arg6 = 0 to 5 iter_args(%arg7 = %arg5) -> (i32) {
          %8 = arith.index_cast %arg6 : index to i32
          %9 = arith.addi %arg7, %8 : i32
          affine.yield %9 : i32
        }
        taskflow.yield %7 : i32
      }) : (i32) -> i32
      %1 = taskflow.channel %data_outs : i32 -> i32
      %control_outs, %data_outs_1 = "taskflow.task"(%arg4) <{operandSegmentSizes = array<i32: 0, 1>, resultSegmentSizes = array<i32: 1, 1>, task_name = "Controller_1"}> ({
      ^bb0(%arg5: i32):
        affine.for %arg6 = 0 to 4 {
          %7 = arith.index_cast %arg6 : index to i32
          %8 = arith.muli %7, %arg5 : i32
          taskflow.emit %arg6, %8 : index, i32
        }
        taskflow.yield
      }) : (i32) -> (!taskflow.packet<index>, i32)
      %2 = taskflow.channel %data_outs_1 : i32 -> i32
      %3 = taskflow.channel %data_outs_1 : i32 -> i32
      %4 = taskflow.drive %control_outs : !taskflow.packet<index> -> !taskflow.packet<index>
      %5 = taskflow.drive %control_outs : !taskflow.packet<index> -> !taskflow.packet<index>
      %data_outs_2 = "taskflow.task"(%5, %3, %arg1) <{operandSegmentSizes = array<i32: 1, 2>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_2"}> ({
      ^bb0(%arg5: index, %arg6: i32, %arg7: memref<4x8xi32>):
        affine.for %arg8 = 0 to 8 {
          %7 = arith.index_cast %arg8 : index to i32
          %8 = arith.addi %arg6, %7 : i32
          memref.store %8, %arg7[%arg5, %arg8] : memref<4x8xi32>
        }
        taskflow.yield %arg7 : memref<4x8xi32>
      }) : (!taskflow.packet<index>, i32, memref<4x8xi32>) -> memref<4x8xi32>
      %6 = taskflow.channel %data_outs_2 : memref<4x8xi32> -> memref<4x8xi32>
      "taskflow.task"(%4, %2, %6, %1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 1, 5>, resultSegmentSizes = array<i32: 0, 1>, task_name = "Task_3"}> ({
      ^bb0(%arg5: index, %arg6: i32, %arg7: memref<4x8xi32>, %arg8: i32, %arg9: memref<i32>, %arg10: i32):
        affine.for %arg11 = 0 to 8 {
          %7 = memref.load %arg7[%arg5, %arg11] : memref<4x8xi32>
          %8 = arith.addi %7, %arg8 : i32
          %c3 = arith.constant 3 : index
          %9 = arith.cmpi eq, %arg5, %c3 : index
          %c7 = arith.constant 7 : index
          %10 = arith.cmpi eq, %arg11, %c7 : index
          %11 = arith.andi %9, %10 : i1
          scf.if %11 {
            memref.store %8, %arg9[] : memref<i32>
            %12 = arith.muli %8, %arg10 : i32
            memref.store %12, %arg9[] : memref<i32>
          }
        }
        taskflow.yield %arg9 : memref<i32>
      }) : (!taskflow.packet<index>, i32, memref<4x8xi32>, i32, memref<i32>, i32) -> memref<i32>
    } : (i32, memref<4x8xi32>, memref<i32>, i32, i32) -> ()
    %0 = affine.load %alloca[] : memref<i32>
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @_Z21irregularLoopExample1v
// CHECK: taskflow.graph
// CHECK: taskflow.task
// CHECK: taskflow.channel
// CHECK: taskflow.yield