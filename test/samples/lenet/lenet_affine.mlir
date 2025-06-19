#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
module {
  func.func @main(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x10xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense_resource<torch_tensor_10_torch.float32> : tensor<10xf32>
    %cst_0 = arith.constant dense_resource<torch_tensor_10_84_torch.float32> : tensor<10x84xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_84_torch.float32> : tensor<84xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_84_120_torch.float32> : tensor<84x120xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_120_torch.float32> : tensor<120xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_120_400_torch.float32> : tensor<120x400xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_16_6_5_5_torch.float32> : tensor<16x6x5x5xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %cst_7 = arith.constant dense_resource<torch_tensor_6_3_5_5_torch.float32> : tensor<6x3x5x5xf32>
    %0 = bufferization.to_memref %arg0 : memref<1x3x32x32xf32>
    %1 = bufferization.to_memref %cst_7 : memref<6x3x5x5xf32>
    %2 = bufferization.to_memref %cst_5 : memref<16x6x5x5xf32>
    %3 = bufferization.to_memref %cst_4 : memref<120x400xf32>
    %4 = bufferization.to_memref %cst_3 : memref<120xf32>
    %5 = bufferization.to_memref %cst_2 : memref<84x120xf32>
    %6 = bufferization.to_memref %cst_1 : memref<84xf32>
    %7 = bufferization.to_memref %cst_0 : memref<10x84xf32>
    %8 = bufferization.to_memref %cst : memref<10xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x6x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 6 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            affine.store %cst_6, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x6x14x14xf32>
          }
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x6x14x14xf32>
    memref.copy %alloc, %alloc_8 : memref<1x6x14x14xf32> to memref<1x6x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 6 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 3 {
              affine.for %arg6 = 0 to 5 {
                affine.for %arg7 = 0 to 5 {
                  %12 = affine.apply #map(%arg3, %arg6)
                  %13 = affine.apply #map(%arg4, %arg7)
                  %14 = affine.load %0[%arg1, %arg5, %12, %13] : memref<1x3x32x32xf32>
                  %15 = affine.load %1[%arg2, %arg5, %arg6, %arg7] : memref<6x3x5x5xf32>
                  %16 = affine.load %alloc_8[%arg1, %arg2, %arg3, %arg4] : memref<1x6x14x14xf32>
                  %17 = arith.mulf %14, %15 : f32
                  %18 = arith.addf %16, %17 : f32
                  affine.store %18, %alloc_8[%arg1, %arg2, %arg3, %arg4] : memref<1x6x14x14xf32>
                }
              }
            }
          }
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x6x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 6 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %12 = affine.load %alloc_8[%c0, %arg2, %arg3, %arg4] : memref<1x6x14x14xf32>
            %13 = arith.cmpf ugt, %12, %cst_6 : f32
            %14 = arith.select %13, %12, %cst_6 : f32
            affine.store %14, %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x6x14x14xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x16x5x5xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 5 {
          affine.for %arg4 = 0 to 5 {
            affine.store %cst_6, %alloc_10[%arg1, %arg2, %arg3, %arg4] : memref<1x16x5x5xf32>
          }
        }
      }
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x16x5x5xf32>
    memref.copy %alloc_10, %alloc_11 : memref<1x16x5x5xf32> to memref<1x16x5x5xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 5 {
          affine.for %arg4 = 0 to 5 {
            affine.for %arg5 = 0 to 6 {
              affine.for %arg6 = 0 to 5 {
                affine.for %arg7 = 0 to 5 {
                  %12 = affine.apply #map(%arg3, %arg6)
                  %13 = affine.apply #map(%arg4, %arg7)
                  %14 = affine.load %alloc_9[%arg1, %arg5, %12, %13] : memref<1x6x14x14xf32>
                  %15 = affine.load %2[%arg2, %arg5, %arg6, %arg7] : memref<16x6x5x5xf32>
                  %16 = affine.load %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x16x5x5xf32>
                  %17 = arith.mulf %14, %15 : f32
                  %18 = arith.addf %16, %17 : f32
                  affine.store %18, %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x16x5x5xf32>
                }
              }
            }
          }
        }
      }
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x16x5x5xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 5 {
          affine.for %arg4 = 0 to 5 {
            %12 = affine.load %alloc_11[%c0, %arg2, %arg3, %arg4] : memref<1x16x5x5xf32>
            %13 = arith.cmpf ugt, %12, %cst_6 : f32
            %14 = arith.select %13, %12, %cst_6 : f32
            affine.store %14, %alloc_12[%arg1, %arg2, %arg3, %arg4] : memref<1x16x5x5xf32>
          }
        }
      }
    }
    %9 = bufferization.to_tensor %alloc_12 : memref<1x16x5x5xf32>
    %collapsed = tensor.collapse_shape %9 [[0], [1, 2, 3]] : tensor<1x16x5x5xf32> into tensor<1x400xf32>
    %10 = bufferization.to_memref %collapsed : memref<1x400xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<400x120xf32>
    affine.for %arg1 = 0 to 120 {
      affine.for %arg2 = 0 to 400 {
        %12 = affine.load %3[%arg1, %arg2] : memref<120x400xf32>
        affine.store %12, %alloc_13[%arg2, %arg1] : memref<400x120xf32>
      }
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x120xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 120 {
        affine.store %cst_6, %alloc_14[%arg1, %arg2] : memref<1x120xf32>
      }
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x120xf32>
    memref.copy %alloc_14, %alloc_15 : memref<1x120xf32> to memref<1x120xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 120 {
        affine.for %arg3 = 0 to 400 {
          %12 = affine.load %10[%arg1, %arg3] : memref<1x400xf32>
          %13 = affine.load %alloc_13[%arg3, %arg2] : memref<400x120xf32>
          %14 = affine.load %alloc_15[%arg1, %arg2] : memref<1x120xf32>
          %15 = arith.mulf %12, %13 : f32
          %16 = arith.addf %14, %15 : f32
          affine.store %16, %alloc_15[%arg1, %arg2] : memref<1x120xf32>
        }
      }
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x120xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 120 {
        %12 = affine.load %alloc_15[%c0, %arg2] : memref<1x120xf32>
        %13 = affine.load %4[%arg2] : memref<120xf32>
        %14 = arith.addf %12, %13 : f32
        affine.store %14, %alloc_16[%arg1, %arg2] : memref<1x120xf32>
      }
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x120xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 120 {
        %12 = affine.load %alloc_16[%c0, %arg2] : memref<1x120xf32>
        %13 = arith.cmpf ugt, %12, %cst_6 : f32
        %14 = arith.select %13, %12, %cst_6 : f32
        affine.store %14, %alloc_17[%arg1, %arg2] : memref<1x120xf32>
      }
    }
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<120x84xf32>
    affine.for %arg1 = 0 to 84 {
      affine.for %arg2 = 0 to 120 {
        %12 = affine.load %5[%arg1, %arg2] : memref<84x120xf32>
        affine.store %12, %alloc_18[%arg2, %arg1] : memref<120x84xf32>
      }
    }
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x84xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 84 {
        affine.store %cst_6, %alloc_19[%arg1, %arg2] : memref<1x84xf32>
      }
    }
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<1x84xf32>
    memref.copy %alloc_19, %alloc_20 : memref<1x84xf32> to memref<1x84xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 84 {
        affine.for %arg3 = 0 to 120 {
          %12 = affine.load %alloc_17[%arg1, %arg3] : memref<1x120xf32>
          %13 = affine.load %alloc_18[%arg3, %arg2] : memref<120x84xf32>
          %14 = affine.load %alloc_20[%arg1, %arg2] : memref<1x84xf32>
          %15 = arith.mulf %12, %13 : f32
          %16 = arith.addf %14, %15 : f32
          affine.store %16, %alloc_20[%arg1, %arg2] : memref<1x84xf32>
        }
      }
    }
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<1x84xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 84 {
        %12 = affine.load %alloc_20[%c0, %arg2] : memref<1x84xf32>
        %13 = affine.load %6[%arg2] : memref<84xf32>
        %14 = arith.addf %12, %13 : f32
        affine.store %14, %alloc_21[%arg1, %arg2] : memref<1x84xf32>
      }
    }
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<1x84xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 84 {
        %12 = affine.load %alloc_21[%c0, %arg2] : memref<1x84xf32>
        %13 = arith.cmpf ugt, %12, %cst_6 : f32
        %14 = arith.select %13, %12, %cst_6 : f32
        affine.store %14, %alloc_22[%arg1, %arg2] : memref<1x84xf32>
      }
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<84x10xf32>
    affine.for %arg1 = 0 to 10 {
      affine.for %arg2 = 0 to 84 {
        %12 = affine.load %7[%arg1, %arg2] : memref<10x84xf32>
        affine.store %12, %alloc_23[%arg2, %arg1] : memref<84x10xf32>
      }
    }
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        affine.store %cst_6, %alloc_24[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    memref.copy %alloc_24, %alloc_25 : memref<1x10xf32> to memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        affine.for %arg3 = 0 to 84 {
          %12 = affine.load %alloc_22[%arg1, %arg3] : memref<1x84xf32>
          %13 = affine.load %alloc_23[%arg3, %arg2] : memref<84x10xf32>
          %14 = affine.load %alloc_25[%arg1, %arg2] : memref<1x10xf32>
          %15 = arith.mulf %12, %13 : f32
          %16 = arith.addf %14, %15 : f32
          affine.store %16, %alloc_25[%arg1, %arg2] : memref<1x10xf32>
        }
      }
    }
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<1x10xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %12 = affine.load %alloc_25[%c0, %arg2] : memref<1x10xf32>
        %13 = affine.load %8[%arg2] : memref<10xf32>
        %14 = arith.addf %12, %13 : f32
        affine.store %14, %alloc_26[%arg1, %arg2] : memref<1x10xf32>
      }
    }
    %11 = bufferization.to_tensor %alloc_26 : memref<1x10xf32>
    return %11 : tensor<1x10xf32>
  }
}

