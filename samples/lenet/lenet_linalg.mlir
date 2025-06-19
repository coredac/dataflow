#map = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x10xf32> {
    %cst = arith.constant dense_resource<torch_tensor_6_3_5_5_torch.float32> : tensor<6x3x5x5xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_16_6_5_5_torch.float32> : tensor<16x6x5x5xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_120_400_torch.float32> : tensor<120x400xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_120_torch.float32> : tensor<120xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_84_120_torch.float32> : tensor<84x120xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_84_torch.float32> : tensor<84xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_10_84_torch.float32> : tensor<10x84xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_10_torch.float32> : tensor<10xf32>
    %0 = tensor.empty() : tensor<1x6x14x14xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%arg0, %cst : tensor<1x3x32x32xf32>, tensor<6x3x5x5xf32>) outs(%1 : tensor<1x6x14x14xf32>) -> tensor<1x6x14x14xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x6x14x14xf32>) outs(%0 : tensor<1x6x14x14xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.cmpf ugt, %in, %cst_0 : f32
      %26 = arith.select %25, %in, %cst_0 : f32
      linalg.yield %26 : f32
    } -> tensor<1x6x14x14xf32>
    %4 = tensor.empty() : tensor<1x16x5x5xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
    %6 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %cst_1 : tensor<1x6x14x14xf32>, tensor<16x6x5x5xf32>) outs(%5 : tensor<1x16x5x5xf32>) -> tensor<1x16x5x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<1x16x5x5xf32>) outs(%4 : tensor<1x16x5x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.cmpf ugt, %in, %cst_0 : f32
      %26 = arith.select %25, %in, %cst_0 : f32
      linalg.yield %26 : f32
    } -> tensor<1x16x5x5xf32>
    %collapsed = tensor.collapse_shape %7 [[0], [1, 2, 3]] : tensor<1x16x5x5xf32> into tensor<1x400xf32>
    %8 = tensor.empty() : tensor<400x120xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<120x400xf32>) outs(%8 : tensor<400x120xf32>) permutation = [1, 0] 
    %9 = tensor.empty() : tensor<1x120xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<1x120xf32>) -> tensor<1x120xf32>
    %11 = linalg.matmul ins(%collapsed, %transposed : tensor<1x400xf32>, tensor<400x120xf32>) outs(%10 : tensor<1x120xf32>) -> tensor<1x120xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%11, %cst_3 : tensor<1x120xf32>, tensor<120xf32>) outs(%9 : tensor<1x120xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %25 = arith.addf %in, %in_10 : f32
      linalg.yield %25 : f32
    } -> tensor<1x120xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map4], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<1x120xf32>) outs(%9 : tensor<1x120xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.cmpf ugt, %in, %cst_0 : f32
      %26 = arith.select %25, %in, %cst_0 : f32
      linalg.yield %26 : f32
    } -> tensor<1x120xf32>
    %14 = tensor.empty() : tensor<120x84xf32>
    %transposed_8 = linalg.transpose ins(%cst_4 : tensor<84x120xf32>) outs(%14 : tensor<120x84xf32>) permutation = [1, 0] 
    %15 = tensor.empty() : tensor<1x84xf32>
    %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<1x84xf32>) -> tensor<1x84xf32>
    %17 = linalg.matmul ins(%13, %transposed_8 : tensor<1x120xf32>, tensor<120x84xf32>) outs(%16 : tensor<1x84xf32>) -> tensor<1x84xf32>
    %18 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%17, %cst_5 : tensor<1x84xf32>, tensor<84xf32>) outs(%15 : tensor<1x84xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %25 = arith.addf %in, %in_10 : f32
      linalg.yield %25 : f32
    } -> tensor<1x84xf32>
    %19 = linalg.generic {indexing_maps = [#map2, #map4], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<1x84xf32>) outs(%15 : tensor<1x84xf32>) {
    ^bb0(%in: f32, %out: f32):
      %25 = arith.cmpf ugt, %in, %cst_0 : f32
      %26 = arith.select %25, %in, %cst_0 : f32
      linalg.yield %26 : f32
    } -> tensor<1x84xf32>
    %20 = tensor.empty() : tensor<84x10xf32>
    %transposed_9 = linalg.transpose ins(%cst_6 : tensor<10x84xf32>) outs(%20 : tensor<84x10xf32>) permutation = [1, 0] 
    %21 = tensor.empty() : tensor<1x10xf32>
    %22 = linalg.fill ins(%cst_0 : f32) outs(%21 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %23 = linalg.matmul ins(%19, %transposed_9 : tensor<1x84xf32>, tensor<84x10xf32>) outs(%22 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %24 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%23, %cst_7 : tensor<1x10xf32>, tensor<10xf32>) outs(%21 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %in_10: f32, %out: f32):
      %25 = arith.addf %in, %in_10 : f32
      linalg.yield %25 : f32
    } -> tensor<1x10xf32>
    return %24 : tensor<1x10xf32>
  }
}