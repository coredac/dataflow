module attributes {torch.debug_module_name = "SimpleResNetBlock"} {
  func.func @forward(%arg0: tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32> {
    %0 = "tosa.const"() <{value = dense<"0x7BEEA13C"> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
    %1 = "tosa.const"() <{value = dense<"0x8B9878BC"> : tensor<64x64x3x3xf32>}> : () -> tensor<64x64x3x3xf32>
    %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %3 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %4 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %5 = tosa.transpose %arg0, %3 : (tensor<1x64x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x64xf32>
    %6 = tosa.transpose %1, %3 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %7 = tosa.conv2d %5, %6, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
    %8 = tosa.transpose %7, %4 : (tensor<1x8x8x64xf32>, tensor<4xi32>) -> tensor<1x64x8x8xf32>
    %9 = tosa.clamp %8 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %10 = tosa.transpose %9, %3 : (tensor<1x64x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x64xf32>
    %11 = tosa.transpose %0, %3 : (tensor<64x64x3x3xf32>, tensor<4xi32>) -> tensor<64x3x3x64xf32>
    %12 = tosa.conv2d %10, %11, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x8x8x64xf32>
    %13 = tosa.transpose %12, %4 : (tensor<1x8x8x64xf32>, tensor<4xi32>) -> tensor<1x64x8x8xf32>
    %14 = tosa.add %13, %arg0 : (tensor<1x64x8x8xf32>, tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    %15 = tosa.clamp %14 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x8x8xf32>) -> tensor<1x64x8x8xf32>
    return %15 : tensor<1x64x8x8xf32>
  }
}
