#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, 0, d3, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map8 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map9 = affine_map<(d0, d1, d2) -> (d2)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map13 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map14 = affine_map<() -> ()>
#map15 = affine_map<(d0, d1, d2, d3) -> ()>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map17 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>
#map18 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @main(%arg0: tensor<1x512x768xf32>, %arg1: tensor<1x128xi64>, %arg2: tensor<1x128xi64>) -> tensor<1x128x768xf32> {
    %cst = arith.constant dense_resource<torch_tensor_30522_768_torch.float32> : tensor<30522x768xf32>
    %c0_i64 = arith.constant 0 : i64
    %c3 = arith.constant 3 : index
    %c30522 = arith.constant 30522 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant 7.670000e+02 : f64
    %cst_4 = arith.constant 0.79788456080286541 : f64
    %cst_5 = arith.constant 4.471500e-02 : f64
    %cst_6 = arith.constant 9.9999999999999995E-7 : f64
    %cst_7 = arith.constant 7.680000e+02 : f32
    %cst_8 = arith.constant 7.680000e+02 : f64
    %cst_9 = arith.constant 8.000000e+00 : f32
    %cst_10 = arith.constant 5.000000e-01 : f32
    %cst_11 = arith.constant 3.000000e+00 : f32
    %cst_12 = arith.constant 1.000000e+00 : f32
    %cst_13 = arith.constant dense_resource<torch_tensor_3_768_torch.float32> : tensor<3x768xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_768_torch.float32> : tensor<768xf32>
    %cst_15 = arith.constant dense_resource<torch_tensor_768_torch.float32_1> : tensor<768xf32>
    %cst_16 = arith.constant dense_resource<torch_tensor_768_768_torch.float32> : tensor<768x768xf32>
    %cst_17 = arith.constant dense_resource<torch_tensor_768_torch.float32_2> : tensor<768xf32>
    %cst_18 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_1> : tensor<768x768xf32>
    %cst_19 = arith.constant dense_resource<torch_tensor_768_torch.float32_3> : tensor<768xf32>
    %cst_20 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_2> : tensor<768x768xf32>
    %cst_21 = arith.constant dense_resource<torch_tensor_768_torch.float32_4> : tensor<768xf32>
    %cst_22 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_3> : tensor<768x768xf32>
    %cst_23 = arith.constant dense_resource<torch_tensor_768_torch.float32_5> : tensor<768xf32>
    %cst_24 = arith.constant dense_resource<torch_tensor_768_torch.float32_6> : tensor<768xf32>
    %cst_25 = arith.constant dense_resource<torch_tensor_768_torch.float32_7> : tensor<768xf32>
    %cst_26 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32> : tensor<3072x768xf32>
    %cst_27 = arith.constant dense_resource<torch_tensor_3072_torch.float32> : tensor<3072xf32>
    %cst_28 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32> : tensor<768x3072xf32>
    %cst_29 = arith.constant dense_resource<torch_tensor_768_torch.float32_8> : tensor<768xf32>
    %cst_30 = arith.constant dense_resource<torch_tensor_768_torch.float32_9> : tensor<768xf32>
    %cst_31 = arith.constant dense_resource<torch_tensor_768_torch.float32_10> : tensor<768xf32>
    %cst_32 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_4> : tensor<768x768xf32>
    %cst_33 = arith.constant dense_resource<torch_tensor_768_torch.float32_11> : tensor<768xf32>
    %cst_34 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_5> : tensor<768x768xf32>
    %cst_35 = arith.constant dense_resource<torch_tensor_768_torch.float32_12> : tensor<768xf32>
    %cst_36 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_6> : tensor<768x768xf32>
    %cst_37 = arith.constant dense_resource<torch_tensor_768_torch.float32_13> : tensor<768xf32>
    %cst_38 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_7> : tensor<768x768xf32>
    %cst_39 = arith.constant dense_resource<torch_tensor_768_torch.float32_14> : tensor<768xf32>
    %cst_40 = arith.constant dense_resource<torch_tensor_768_torch.float32_15> : tensor<768xf32>
    %cst_41 = arith.constant dense_resource<torch_tensor_768_torch.float32_16> : tensor<768xf32>
    %cst_42 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_1> : tensor<3072x768xf32>
    %cst_43 = arith.constant dense_resource<torch_tensor_3072_torch.float32_1> : tensor<3072xf32>
    %cst_44 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_1> : tensor<768x3072xf32>
    %cst_45 = arith.constant dense_resource<torch_tensor_768_torch.float32_17> : tensor<768xf32>
    %cst_46 = arith.constant dense<-1.000000e+09> : tensor<f64>
    %0 = tensor.empty() : tensor<1x128xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x128xi64>) outs(%0 : tensor<1x128xi1>) {
    ^bb0(%in: i64, %out: i1):
      %195 = arith.cmpi sgt, %in, %c0_i64 : i64
      linalg.yield %195 : i1
    } -> tensor<1x128xi1>
    %expanded = tensor.expand_shape %1 [[0, 1], [2, 3, 4, 5]] : tensor<1x128xi1> into tensor<1x1x1x1x1x128xi1>
    %2 = tensor.empty() : tensor<1x1x128x1x1x128xi1>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x1x1x1x1x128xi1>) outs(%2 : tensor<1x1x128x1x1x128xi1>) {
    ^bb0(%in: i1, %out: i1):
      linalg.yield %in : i1
    } -> tensor<1x1x128x1x1x128xi1>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2], [3, 4, 5]] : tensor<1x1x128x1x1x128xi1> into tensor<1x128x128xi1>
    %expanded_47 = tensor.expand_shape %collapsed [[0], [1, 2], [3]] : tensor<1x128x128xi1> into tensor<1x1x128x128xi1>
    %4 = tensor.empty() : tensor<1x128x768xf32>
    %5 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x128xi64>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %195 = arith.index_cast %in : i64 to index
      %196 = linalg.index 2 : index
      %197 = arith.cmpi slt, %195, %c30522 : index
      cf.assert %197, "index must be smaller than dim size"
      %198 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %198, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst[%195, %196] : tensor<30522x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x128x768xf32>
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x512x768xf32> to tensor<1x128x768xf32>
    %6 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %extracted_slice : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %7 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x128xi64>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %195 = arith.index_cast %in : i64 to index
      %196 = linalg.index 2 : index
      %197 = arith.cmpi slt, %195, %c3 : index
      cf.assert %197, "index must be smaller than dim size"
      %198 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %198, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_13[%195, %196] : tensor<3x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x128x768xf32>
    %8 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %7 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %9 = tensor.empty() : tensor<1x128x1xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
    %11 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8 : tensor<1x128x768xf32>) outs(%10 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %out : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %12 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.divf %in, %cst_7 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %13 = tensor.empty() : tensor<1x128x768xf64>
    %14 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<1x128x768xf32>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %195 = arith.extf %in : f32 to f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %15 = tensor.empty() : tensor<1x128x1xf64>
    %16 = linalg.fill ins(%cst_1 : f64) outs(%15 : tensor<1x128x1xf64>) -> tensor<1x128x1xf64>
    %17 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%14 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %18 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_8 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %19 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %18 : tensor<1x128x768xf64>, tensor<1x128x1xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.subf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %20 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %19 : tensor<1x128x768xf64>, tensor<1x128x768xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.mulf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %21 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%20 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %22 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_3 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %23 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<1x128x1xf64>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %195 = arith.truncf %in : f64 to f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %24 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.sqrt %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %25 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %12 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.subf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %26 = linalg.generic {indexing_maps = [#map9, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_14, %25 : tensor<768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.mulf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %27 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_6 : f64 to f32
      %196 = arith.addf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x1xf32>
    %28 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %27 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.divf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %29 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %cst_15 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %30 = tensor.empty() : tensor<768x768xf32>
    %transposed = linalg.transpose ins(%cst_16 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %31 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29 : tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %32 = tensor.empty() : tensor<1x768x768xf32>
    %33 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %34 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %35 = linalg.batch_matmul ins(%31, %33 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %36 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %cst_17 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %expanded_48 = tensor.expand_shape %36 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %37 = tensor.empty() : tensor<1x12x128x64xf32>
    %transposed_49 = linalg.transpose ins(%expanded_48 : tensor<1x128x12x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_50 = linalg.transpose ins(%cst_18 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %38 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_50 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %39 = linalg.batch_matmul ins(%31, %38 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %40 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %cst_19 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %expanded_51 = tensor.expand_shape %40 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_52 = linalg.transpose ins(%cst_20 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %41 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_52 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %42 = linalg.batch_matmul ins(%31, %41 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %43 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %cst_21 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %expanded_53 = tensor.expand_shape %43 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_54 = linalg.transpose ins(%expanded_53 : tensor<1x128x12x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %44 = tensor.empty() : tensor<1x12x64x128xf32>
    %transposed_55 = linalg.transpose ins(%expanded_51 : tensor<1x128x12x64xf32>) outs(%44 : tensor<1x12x64x128xf32>) permutation = [0, 2, 3, 1] 
    %45 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_49 : tensor<1x12x128x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %46 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_55 : tensor<1x12x64x128xf32>) outs(%44 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_56 = tensor.collapse_shape %45 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_57 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %47 = tensor.empty() : tensor<12x128x128xf32>
    %48 = linalg.fill ins(%cst_0 : f32) outs(%47 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %49 = linalg.batch_matmul ins(%collapsed_56, %collapsed_57 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%48 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_58 = tensor.expand_shape %49 [[0, 1], [2], [3]] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %50 = tensor.empty() : tensor<1x12x128x128xf32>
    %51 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_58 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.divf %in, %cst_9 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %52 = tensor.empty() : tensor<1x1x128x128xi1>
    %53 = linalg.generic {indexing_maps = [#map13, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_47 : tensor<1x1x128x128xi1>) outs(%52 : tensor<1x1x128x128xi1>) {
    ^bb0(%in: i1, %out: i1):
      %195 = arith.extui %in : i1 to i64
      %196 = arith.cmpi eq, %195, %c0_i64 : i64
      linalg.yield %196 : i1
    } -> tensor<1x1x128x128xi1>
    %54 = tensor.empty() : tensor<f32>
    %55 = linalg.generic {indexing_maps = [#map14, #map14], iterator_types = []} ins(%cst_46 : tensor<f64>) outs(%54 : tensor<f32>) {
    ^bb0(%in: f64, %out: f32):
      %195 = arith.truncf %in : f64 to f32
      linalg.yield %195 : f32
    } -> tensor<f32>
    %56 = linalg.generic {indexing_maps = [#map13, #map15, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %55, %51 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_89: f32, %in_90: f32, %out: f32):
      %195 = arith.select %in, %in_89, %in_90 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %57 = tensor.empty() : tensor<1x12x128xi64>
    %58 = linalg.fill ins(%c0_i64 : i64) outs(%57 : tensor<1x12x128xi64>) -> tensor<1x12x128xi64>
    %59 = tensor.empty() : tensor<1x12x128xf32>
    %60 = linalg.fill ins(%cst_2 : f32) outs(%59 : tensor<1x12x128xf32>) -> tensor<1x12x128xf32>
    %61:2 = linalg.generic {indexing_maps = [#map12, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%56 : tensor<1x12x128x128xf32>) outs(%60, %58 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_89: i64):
      %195 = linalg.index 3 : index
      %196 = arith.index_cast %195 : index to i64
      %197 = arith.maximumf %in, %out : f32
      %198 = arith.cmpf ogt, %in, %out : f32
      %199 = arith.select %198, %196, %out_89 : i64
      linalg.yield %197, %199 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_59 = tensor.expand_shape %61#0 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %62 = linalg.generic {indexing_maps = [#map11, #map17, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%56, %expanded_59 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.subf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %63 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.exp %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %64 = tensor.empty() : tensor<1x12x128x1xf32>
    %65 = linalg.fill ins(%cst_0 : f32) outs(%64 : tensor<1x12x128x1xf32>) -> tensor<1x12x128x1xf32>
    %66 = linalg.generic {indexing_maps = [#map12, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%63 : tensor<1x12x128x128xf32>) outs(%65 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %out : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x1xf32>
    %67 = linalg.generic {indexing_maps = [#map11, #map17, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63, %66 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.divf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %68 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %69 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_54 : tensor<1x12x128x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %collapsed_60 = tensor.collapse_shape %68 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %collapsed_61 = tensor.collapse_shape %69 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %70 = tensor.empty() : tensor<12x128x64xf32>
    %71 = linalg.fill ins(%cst_0 : f32) outs(%70 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %72 = linalg.batch_matmul ins(%collapsed_60, %collapsed_61 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%71 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_62 = tensor.expand_shape %72 [[0, 1], [2], [3]]: tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %73 = tensor.empty() : tensor<1x128x12x64xf32>
    %transposed_63 = linalg.transpose ins(%expanded_62 : tensor<1x12x128x64xf32>) outs(%73 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_64 = tensor.collapse_shape %transposed_63 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %transposed_65 = linalg.transpose ins(%cst_22 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %74 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_64 : tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %75 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_65 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %76 = linalg.batch_matmul ins(%74, %75 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %77 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %cst_23 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %78 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %77 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %79 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x128x768xf32>) outs(%10 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %out : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %80 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.divf %in, %cst_7 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %81 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<1x128x768xf32>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %195 = arith.extf %in : f32 to f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %82 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %83 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_8 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %84 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %83 : tensor<1x128x768xf64>, tensor<1x128x1xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.subf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %85 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %84 : tensor<1x128x768xf64>, tensor<1x128x768xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.mulf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %86 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %87 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_3 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %88 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<1x128x1xf64>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %195 = arith.truncf %in : f64 to f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %89 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.sqrt %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %90 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78, %80 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.subf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %91 = linalg.generic {indexing_maps = [#map9, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_24, %90 : tensor<768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.mulf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %92 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_6 : f64 to f32
      %196 = arith.addf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x1xf32>
    %93 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %92 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.divf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %94 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %cst_25 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %95 = tensor.empty() : tensor<768x3072xf32>
    %transposed_66 = linalg.transpose ins(%cst_26 : tensor<3072x768xf32>) outs(%95 : tensor<768x3072xf32>) permutation = [1, 0] 
    %96 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94 : tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %97 = tensor.empty() : tensor<1x768x3072xf32>
    %98 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_66 : tensor<768x3072xf32>) outs(%97 : tensor<1x768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x3072xf32>
    %99 = tensor.empty() : tensor<1x128x3072xf32>
    %100 = linalg.fill ins(%cst_0 : f32) outs(%99 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %101 = linalg.batch_matmul ins(%96, %98 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%100 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %102 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %cst_27 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %103 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.mulf %in, %cst_10 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %104 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.powf %in, %cst_11 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %105 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_5 : f64 to f32
      %196 = arith.mulf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x3072xf32>
    %106 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %105 : tensor<1x128x3072xf32>, tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %107 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_4 : f64 to f32
      %196 = arith.mulf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x3072xf32>
    %108 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%107 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.tanh %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %109 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %cst_12 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %110 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%103, %109 : tensor<1x128x3072xf32>, tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.mulf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %111 = tensor.empty() : tensor<3072x768xf32>
    %transposed_67 = linalg.transpose ins(%cst_28 : tensor<768x3072xf32>) outs(%111 : tensor<3072x768xf32>) permutation = [1, 0] 
    %112 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%110 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %113 = tensor.empty() : tensor<1x3072x768xf32>
    %114 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_67 : tensor<3072x768xf32>) outs(%113 : tensor<1x3072x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3072x768xf32>
    %115 = linalg.batch_matmul ins(%112, %114 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %116 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %cst_29 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %117 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78, %116 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %118 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%117 : tensor<1x128x768xf32>) outs(%10 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %out : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %119 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.divf %in, %cst_7 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %120 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117 : tensor<1x128x768xf32>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %195 = arith.extf %in : f32 to f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %121 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %122 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%121 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_8 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %123 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<1x128x768xf64>, tensor<1x128x1xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.subf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %124 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %123 : tensor<1x128x768xf64>, tensor<1x128x768xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.mulf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %125 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%124 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %126 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_3 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %127 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%126 : tensor<1x128x1xf64>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %195 = arith.truncf %in : f64 to f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %128 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.sqrt %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %129 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %119 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.subf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %130 = linalg.generic {indexing_maps = [#map9, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_30, %129 : tensor<768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.mulf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %131 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_6 : f64 to f32
      %196 = arith.addf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x1xf32>
    %132 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %131 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.divf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %133 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%132, %cst_31 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %transposed_68 = linalg.transpose ins(%cst_32 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %134 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133 : tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %135 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_68 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %136 = linalg.batch_matmul ins(%134, %135 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %137 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%136, %cst_33 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %expanded_69 = tensor.expand_shape %137 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_70 = linalg.transpose ins(%expanded_69 : tensor<1x128x12x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_71 = linalg.transpose ins(%cst_34 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %138 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_71 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %139 = linalg.batch_matmul ins(%134, %138 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %140 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%139, %cst_35 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %expanded_72 = tensor.expand_shape %140 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_73 = linalg.transpose ins(%cst_36 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %141 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_73 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %142 = linalg.batch_matmul ins(%134, %141 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %143 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142, %cst_37 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %expanded_74 = tensor.expand_shape %143 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %transposed_75 = linalg.transpose ins(%expanded_74 : tensor<1x128x12x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) permutation = [0, 2, 1, 3] 
    %transposed_76 = linalg.transpose ins(%expanded_72 : tensor<1x128x12x64xf32>) outs(%44 : tensor<1x12x64x128xf32>) permutation = [0, 2, 3, 1] 
    %144 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_70 : tensor<1x12x128x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %145 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_76 : tensor<1x12x64x128xf32>) outs(%44 : tensor<1x12x64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x64x128xf32>
    %collapsed_77 = tensor.collapse_shape %144 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %collapsed_78 = tensor.collapse_shape %145 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %146 = linalg.batch_matmul ins(%collapsed_77, %collapsed_78 : tensor<12x128x64xf32>, tensor<12x64x128xf32>) outs(%48 : tensor<12x128x128xf32>) -> tensor<12x128x128xf32>
    %expanded_79 = tensor.expand_shape %146 [[0, 1], [2], [3]] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %147 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_79 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.divf %in, %cst_9 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %148 = linalg.generic {indexing_maps = [#map13, #map15, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %55, %147 : tensor<1x1x128x128xi1>, tensor<f32>, tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: i1, %in_89: f32, %in_90: f32, %out: f32):
      %195 = arith.select %in, %in_89, %in_90 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %149:2 = linalg.generic {indexing_maps = [#map12, #map16, #map16], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%148 : tensor<1x12x128x128xf32>) outs(%60, %58 : tensor<1x12x128xf32>, tensor<1x12x128xi64>) {
    ^bb0(%in: f32, %out: f32, %out_89: i64):
      %195 = linalg.index 3 : index
      %196 = arith.index_cast %195 : index to i64
      %197 = arith.maximumf %in, %out : f32
      %198 = arith.cmpf ogt, %in, %out : f32
      %199 = arith.select %198, %196, %out_89 : i64
      linalg.yield %197, %199 : f32, i64
    } -> (tensor<1x12x128xf32>, tensor<1x12x128xi64>)
    %expanded_80 = tensor.expand_shape %149#0 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %150 = linalg.generic {indexing_maps = [#map11, #map17, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148, %expanded_80 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.subf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %151 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%150 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.exp %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %152 = linalg.generic {indexing_maps = [#map12, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%151 : tensor<1x12x128x128xf32>) outs(%65 : tensor<1x12x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %out : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x1xf32>
    %153 = linalg.generic {indexing_maps = [#map11, #map17, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151, %152 : tensor<1x12x128x128xf32>, tensor<1x12x128x1xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.divf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x12x128x128xf32>
    %154 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%153 : tensor<1x12x128x128xf32>) outs(%50 : tensor<1x12x128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x128xf32>
    %155 = linalg.generic {indexing_maps = [#map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_75 : tensor<1x12x128x64xf32>) outs(%37 : tensor<1x12x128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x128x64xf32>
    %collapsed_81 = tensor.collapse_shape %154 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %collapsed_82 = tensor.collapse_shape %155 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %156 = linalg.batch_matmul ins(%collapsed_81, %collapsed_82 : tensor<12x128x128xf32>, tensor<12x128x64xf32>) outs(%71 : tensor<12x128x64xf32>) -> tensor<12x128x64xf32>
    %expanded_83 = tensor.expand_shape %156 [[0, 1], [2], [3]] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %transposed_84 = linalg.transpose ins(%expanded_83 : tensor<1x12x128x64xf32>) outs(%73 : tensor<1x128x12x64xf32>) permutation = [0, 2, 1, 3] 
    %collapsed_85 = tensor.collapse_shape %transposed_84 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %transposed_86 = linalg.transpose ins(%cst_38 : tensor<768x768xf32>) outs(%30 : tensor<768x768xf32>) permutation = [1, 0] 
    %157 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_85 : tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %158 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_86 : tensor<768x768xf32>) outs(%32 : tensor<1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x768xf32>
    %159 = linalg.batch_matmul ins(%157, %158 : tensor<1x128x768xf32>, tensor<1x768x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %160 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%159, %cst_39 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %161 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %160 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %162 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%161 : tensor<1x128x768xf32>) outs(%10 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %out : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %163 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%162 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.divf %in, %cst_7 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %164 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161 : tensor<1x128x768xf32>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %195 = arith.extf %in : f32 to f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %165 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%164 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %166 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%165 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_8 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %167 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%164, %166 : tensor<1x128x768xf64>, tensor<1x128x1xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.subf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %168 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%167, %167 : tensor<1x128x768xf64>, tensor<1x128x768xf64>) outs(%13 : tensor<1x128x768xf64>) {
    ^bb0(%in: f64, %in_89: f64, %out: f64):
      %195 = arith.mulf %in, %in_89 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x768xf64>
    %169 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "reduction"]} ins(%168 : tensor<1x128x768xf64>) outs(%16 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.addf %in, %out : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %170 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169 : tensor<1x128x1xf64>) outs(%15 : tensor<1x128x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %195 = arith.divf %in, %cst_3 : f64
      linalg.yield %195 : f64
    } -> tensor<1x128x1xf64>
    %171 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%170 : tensor<1x128x1xf64>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %195 = arith.truncf %in : f64 to f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %172 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%171 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.sqrt %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x1xf32>
    %173 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161, %163 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.subf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %174 = linalg.generic {indexing_maps = [#map9, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_40, %173 : tensor<768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.mulf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %175 = linalg.generic {indexing_maps = [#map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172 : tensor<1x128x1xf32>) outs(%9 : tensor<1x128x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_6 : f64 to f32
      %196 = arith.addf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x1xf32>
    %176 = linalg.generic {indexing_maps = [#map6, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174, %175 : tensor<1x128x768xf32>, tensor<1x128x1xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.divf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %177 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%176, %cst_41 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %transposed_87 = linalg.transpose ins(%cst_42 : tensor<3072x768xf32>) outs(%95 : tensor<768x3072xf32>) permutation = [1, 0] 
    %178 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%177 : tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x768xf32>
    %179 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_87 : tensor<768x3072xf32>) outs(%97 : tensor<1x768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x768x3072xf32>
    %180 = linalg.batch_matmul ins(%178, %179 : tensor<1x128x768xf32>, tensor<1x768x3072xf32>) outs(%100 : tensor<1x128x3072xf32>) -> tensor<1x128x3072xf32>
    %181 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180, %cst_43 : tensor<1x128x3072xf32>, tensor<3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %182 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.mulf %in, %cst_10 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %183 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.powf %in, %cst_11 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %184 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%183 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_5 : f64 to f32
      %196 = arith.mulf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x3072xf32>
    %185 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181, %184 : tensor<1x128x3072xf32>, tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %186 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%185 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.truncf %cst_4 : f64 to f32
      %196 = arith.mulf %in, %195 : f32
      linalg.yield %196 : f32
    } -> tensor<1x128x3072xf32>
    %187 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%186 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = math.tanh %in : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %188 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%187 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %195 = arith.addf %in, %cst_12 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %189 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%182, %188 : tensor<1x128x3072xf32>, tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.mulf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x3072xf32>
    %transposed_88 = linalg.transpose ins(%cst_44 : tensor<768x3072xf32>) outs(%111 : tensor<3072x768xf32>) permutation = [1, 0] 
    %190 = linalg.generic {indexing_maps = [#map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%189 : tensor<1x128x3072xf32>) outs(%99 : tensor<1x128x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x128x3072xf32>
    %191 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%transposed_88 : tensor<3072x768xf32>) outs(%113 : tensor<1x3072x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3072x768xf32>
    %192 = linalg.batch_matmul ins(%190, %191 : tensor<1x128x3072xf32>, tensor<1x3072x768xf32>) outs(%34 : tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %193 = linalg.generic {indexing_maps = [#map6, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192, %cst_45 : tensor<1x128x768xf32>, tensor<768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    %194 = linalg.generic {indexing_maps = [#map6, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161, %193 : tensor<1x128x768xf32>, tensor<1x128x768xf32>) outs(%4 : tensor<1x128x768xf32>) {
    ^bb0(%in: f32, %in_89: f32, %out: f32):
      %195 = arith.addf %in, %in_89 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x768xf32>
    return %194 : tensor<1x128x768xf32>
  }
}