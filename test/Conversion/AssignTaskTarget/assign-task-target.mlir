// RUN: mlir-neura-opt %s --assign-task-target -o %S/Output/assign-task-target.mlir.tmp
// RUN: mlir-neura-opt %s --assign-task-target | FileCheck %s

// Test the AssignTaskTarget pass with NeRF modular functions

module {
  // CHECK-LABEL: func.func @ray_sampler_func
  // CHECK-SAME: attributes {target.device = "cpu"}
  func.func @ray_sampler_func(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) 
                           -> tensor<2x16x3xf32> {
    %0 = arith.constant 0.0 : f32
    %1 = tensor.empty() : tensor<2x16x3xf32>
    return %1 : tensor<2x16x3xf32>
  }

  // CHECK-LABEL: func.func @hash_encoder_func
  // CHECK-SAME: attributes {target.device = "doe"}
  func.func @hash_encoder_func(%arg0: tensor<2x16x3xf32>) 
                            -> tensor<2x16x4xf32> {
    %0 = tensor.empty() : tensor<2x16x4xf32>
    return %0 : tensor<2x16x4xf32>
  }

  // CHECK-LABEL: func.func @nerf_mlp_func
  // CHECK-SAME: attributes {target.device = "cgra"}
  func.func @nerf_mlp_func(%arg0: tensor<2x16x4xf32>, %arg1: tensor<2x3xf32>) 
                        -> (tensor<2x16x1xf32>, tensor<2x16x3xf32>) {
    %0 = tensor.empty() : tensor<2x16x1xf32>
    %1 = tensor.empty() : tensor<2x16x3xf32>
    return %0, %1 : tensor<2x16x1xf32>, tensor<2x16x3xf32>
  }

  // CHECK-LABEL: func.func @nerf_forward
  // CHECK-SAME: attributes {target.device = "cpu"}
  func.func @nerf_forward(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) 
                       -> (tensor<2x16x1xf32>, tensor<2x16x3xf32>) {
    %positions = func.call @ray_sampler_func(%arg0, %arg1) 
                 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x16x3xf32>
    
    %encoded = func.call @hash_encoder_func(%positions)
               : (tensor<2x16x3xf32>) -> tensor<2x16x4xf32>
    
    %density, %rgb = func.call @nerf_mlp_func(%encoded, %arg1)
                     : (tensor<2x16x4xf32>, tensor<2x3xf32>) 
                     -> (tensor<2x16x1xf32>, tensor<2x16x3xf32>)
    
    return %density, %rgb : tensor<2x16x1xf32>, tensor<2x16x3xf32>
  }

  // CHECK-LABEL: func.func @generic_sampler
  // CHECK-SAME: attributes {target.device = "cpu"}
  func.func @generic_sampler() {
    return
  }

  // CHECK-LABEL: func.func @custom_encoder
  // CHECK-SAME: attributes {target.device = "doe"}
  func.func @custom_encoder() {
    return
  }

  // CHECK-LABEL: func.func @some_mlp
  // CHECK-SAME: attributes {target.device = "cgra"}
  func.func @some_mlp() {
    return
  }

  // CHECK-LABEL: func.func @unknown_function
  // CHECK-SAME: attributes {target.device = "cpu"}
  func.func @unknown_function() {
    return
  }
}
