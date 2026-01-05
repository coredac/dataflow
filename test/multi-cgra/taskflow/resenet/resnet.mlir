// RUN: cd %S && python resnet.py

// RUN: mlir-taskflow-opt %S/Output/simple_resnet.mlir \
// RUN: --convert-linalg-to-taskflow -o %t-resnet-taskflow.mlir \
// RUN: | FileCheck %s --input-file=%t-resnet-taskflow.mlir

// CHECK:      %2 = taskflow.graph(%arg0, %cst_1, %cst_0, %1, %0, %cst) {
// CHECK-NEXT:   ^bb0(%arg1: tensor<1x64x8x8xf32>, %arg2: f32, %arg3: tensor<64x64x3x3xf32>, %arg4: tensor<1x64x8x8xf32>, %arg5: tensor<1x64x8x8xf32>, %arg6: tensor<64x64x3x3xf32>):
// CHECK-NEXT:     %data_outs = "taskflow.task"(%arg1, %arg2) <{operandSegmentSizes = array<i32: 0, 2>, resultSegmentSizes = array<i32: 0, 1>, task_name = "task_0"}> ({
// CHECK-NEXT:     ^bb0(%arg7: tensor<1x64x8x8xf32>, %arg8: f32):
// CHECK-NEXT:       %padded = tensor.pad %arg7 low[0, 0, 1, 1] high[0, 0, 1, 1] {
// CHECK-NEXT:       ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index):
// CHECK-NEXT:         tensor.yield %arg8 : f32
// CHECK-NEXT:       } : tensor<1x64x8x8xf32> to tensor<1x64x10x10xf32>
// CHECK-NEXT:       taskflow.yield %padded : tensor<1x64x10x10xf32>
// CHECK-NEXT:     }) : (tensor<1x64x8x8xf32>, f32) -> tensor<1x64x10x10xf32>
// CHECK-NEXT:     %3 = taskflow.channel %data_outs : tensor<1x64x10x10xf32> -> tensor<1x64x10x10xf32>
// CHECK-NEXT:     %data_outs_2 = "taskflow.task"(%arg3, %arg4, %3) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "conv2d_1"}> ({
// CHECK-NEXT:     ^bb0(%arg7: tensor<64x64x3x3xf32>, %arg8: tensor<1x64x8x8xf32>, %arg9: tensor<1x64x10x10xf32>):
// CHECK-NEXT:       %9 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%data_outs, %arg7 : tensor<1x64x10x10xf32>, tensor<64x64x3x3xf32>) outs(%arg8 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:       taskflow.yield %9 : tensor<1x64x8x8xf32>
// CHECK-NEXT:     }) : (tensor<64x64x3x3xf32>, tensor<1x64x8x8xf32>, tensor<1x64x10x10xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %4 = taskflow.channel %data_outs_2 : tensor<1x64x8x8xf32> -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %data_outs_3 = "taskflow.task"(%arg5, %arg2, %4) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "generic_2"}> ({
// CHECK-NEXT:     ^bb0(%arg7: tensor<1x64x8x8xf32>, %arg8: f32, %arg9: tensor<1x64x8x8xf32>):
// CHECK-NEXT:       %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%data_outs_2 : tensor<1x64x8x8xf32>) outs(%arg7 : tensor<1x64x8x8xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %10 = arith.cmpf ugt, %in, %arg8 : f32
// CHECK-NEXT:         %11 = arith.select %10, %in, %arg8 : f32
// CHECK-NEXT:         linalg.yield %11 : f32
// CHECK-NEXT:       } -> tensor<1x64x8x8xf32>
// CHECK-NEXT:       taskflow.yield %9 : tensor<1x64x8x8xf32>
// CHECK-NEXT:     }) : (tensor<1x64x8x8xf32>, f32, tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %5 = taskflow.channel %data_outs_3 : tensor<1x64x8x8xf32> -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %data_outs_4 = "taskflow.task"(%arg2, %5) <{operandSegmentSizes = array<i32: 0, 2>, resultSegmentSizes = array<i32: 0, 1>, task_name = "task_3"}> ({
// CHECK-NEXT:     ^bb0(%arg7: f32, %arg8: tensor<1x64x8x8xf32>):
// CHECK-NEXT:       %padded = tensor.pad %data_outs_3 low[0, 0, 1, 1] high[0, 0, 1, 1] {
// CHECK-NEXT:       ^bb0(%arg9: index, %arg10: index, %arg11: index, %arg12: index):
// CHECK-NEXT:         tensor.yield %arg7 : f32
// CHECK-NEXT:       } : tensor<1x64x8x8xf32> to tensor<1x64x10x10xf32>
// CHECK-NEXT:       taskflow.yield %padded : tensor<1x64x10x10xf32>
// CHECK-NEXT:     }) : (f32, tensor<1x64x8x8xf32>) -> tensor<1x64x10x10xf32>
// CHECK-NEXT:     %6 = taskflow.channel %data_outs_4 : tensor<1x64x10x10xf32> -> tensor<1x64x10x10xf32>
// CHECK-NEXT:     %data_outs_5 = "taskflow.task"(%arg6, %arg4, %6) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "conv2d_4"}> ({
// CHECK-NEXT:     ^bb0(%arg7: tensor<64x64x3x3xf32>, %arg8: tensor<1x64x8x8xf32>, %arg9: tensor<1x64x10x10xf32>):
// CHECK-NEXT:       %9 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%data_outs_4, %arg7 : tensor<1x64x10x10xf32>, tensor<64x64x3x3xf32>) outs(%arg8 : tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:       taskflow.yield %9 : tensor<1x64x8x8xf32>
// CHECK-NEXT:     }) : (tensor<64x64x3x3xf32>, tensor<1x64x8x8xf32>, tensor<1x64x10x10xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %7 = taskflow.channel %data_outs_5 : tensor<1x64x8x8xf32> -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %data_outs_6 = "taskflow.task"(%arg1, %arg5, %7) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "generic_5"}> ({
// CHECK-NEXT:     ^bb0(%arg7: tensor<1x64x8x8xf32>, %arg8: tensor<1x64x8x8xf32>, %arg9: tensor<1x64x8x8xf32>):
// CHECK-NEXT:       %9 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%data_outs_5, %arg7 : tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) outs(%arg8 : tensor<1x64x8x8xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:         %10 = arith.addf %in, %in_8 : f32
// CHECK-NEXT:         linalg.yield %10 : f32
// CHECK-NEXT:       } -> tensor<1x64x8x8xf32>
// CHECK-NEXT:       taskflow.yield %9 : tensor<1x64x8x8xf32>
// CHECK-NEXT:     }) : (tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %8 = taskflow.channel %data_outs_6 : tensor<1x64x8x8xf32> -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     %data_outs_7 = "taskflow.task"(%arg5, %arg2, %8) <{operandSegmentSizes = array<i32: 0, 3>, resultSegmentSizes = array<i32: 0, 1>, task_name = "generic_6"}> ({
// CHECK-NEXT:     ^bb0(%arg7: tensor<1x64x8x8xf32>, %arg8: f32, %arg9: tensor<1x64x8x8xf32>):
// CHECK-NEXT:       %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%data_outs_6 : tensor<1x64x8x8xf32>) outs(%arg7 : tensor<1x64x8x8xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %10 = arith.cmpf ugt, %in, %arg8 : f32
// CHECK-NEXT:         %11 = arith.select %10, %in, %arg8 : f32
// CHECK-NEXT:         linalg.yield %11 : f32
// CHECK-NEXT:       } -> tensor<1x64x8x8xf32>
// CHECK-NEXT:       taskflow.yield %9 : tensor<1x64x8x8xf32>
// CHECK-NEXT:     }) : (tensor<1x64x8x8xf32>, f32, tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:     taskflow.return %data_outs_7 : tensor<1x64x8x8xf32>
// CHECK-NEXT:   } : (tensor<1x64x8x8xf32>, f32, tensor<64x64x3x3xf32>, tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x8x8xf32>
// CHECK-NEXT:   return %2 : tensor<1x64x8x8xf32>
// CHECK-NEXT: }