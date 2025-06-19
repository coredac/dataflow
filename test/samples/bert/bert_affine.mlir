module {
  func.func @main(%arg0: tensor<1x512x768xf32>, %arg1: tensor<1x128xi64>, %arg2: tensor<1x128xi64>) -> tensor<1x128x768xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<-1.000000e+09> : tensor<f64>
    %cst_0 = arith.constant dense_resource<torch_tensor_768_torch.float32_17> : tensor<768xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_1> : tensor<768x3072xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_3072_torch.float32_1> : tensor<3072xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_1> : tensor<3072x768xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_768_torch.float32_16> : tensor<768xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_768_torch.float32_15> : tensor<768xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_768_torch.float32_14> : tensor<768xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_7> : tensor<768x768xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_768_torch.float32_13> : tensor<768xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_6> : tensor<768x768xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_768_torch.float32_12> : tensor<768xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_5> : tensor<768x768xf32>
    %cst_12 = arith.constant dense_resource<torch_tensor_768_torch.float32_11> : tensor<768xf32>
    %cst_13 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_4> : tensor<768x768xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_768_torch.float32_10> : tensor<768xf32>
    %cst_15 = arith.constant dense_resource<torch_tensor_768_torch.float32_9> : tensor<768xf32>
    %cst_16 = arith.constant dense_resource<torch_tensor_768_torch.float32_8> : tensor<768xf32>
    %cst_17 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32> : tensor<768x3072xf32>
    %cst_18 = arith.constant dense_resource<torch_tensor_3072_torch.float32> : tensor<3072xf32>
    %cst_19 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32> : tensor<3072x768xf32>
    %cst_20 = arith.constant dense_resource<torch_tensor_768_torch.float32_7> : tensor<768xf32>
    %cst_21 = arith.constant dense_resource<torch_tensor_768_torch.float32_6> : tensor<768xf32>
    %cst_22 = arith.constant dense_resource<torch_tensor_768_torch.float32_5> : tensor<768xf32>
    %cst_23 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_3> : tensor<768x768xf32>
    %cst_24 = arith.constant dense_resource<torch_tensor_768_torch.float32_4> : tensor<768xf32>
    %cst_25 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_2> : tensor<768x768xf32>
    %cst_26 = arith.constant dense_resource<torch_tensor_768_torch.float32_3> : tensor<768xf32>
    %cst_27 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_1> : tensor<768x768xf32>
    %cst_28 = arith.constant dense_resource<torch_tensor_768_torch.float32_2> : tensor<768xf32>
    %cst_29 = arith.constant dense_resource<torch_tensor_768_768_torch.float32> : tensor<768x768xf32>
    %cst_30 = arith.constant dense_resource<torch_tensor_768_torch.float32_1> : tensor<768xf32>
    %cst_31 = arith.constant dense_resource<torch_tensor_768_torch.float32> : tensor<768xf32>
    %cst_32 = arith.constant dense_resource<torch_tensor_3_768_torch.float32> : tensor<3x768xf32>
    %cst_33 = arith.constant 1.000000e+00 : f32
    %cst_34 = arith.constant 3.000000e+00 : f32
    %cst_35 = arith.constant 5.000000e-01 : f32
    %cst_36 = arith.constant 8.000000e+00 : f32
    %cst_37 = arith.constant 7.680000e+02 : f64
    %cst_38 = arith.constant 7.680000e+02 : f32
    %cst_39 = arith.constant 9.9999999999999995E-7 : f64
    %cst_40 = arith.constant 4.471500e-02 : f64
    %cst_41 = arith.constant 0.79788456080286541 : f64
    %cst_42 = arith.constant 7.670000e+02 : f64
    %cst_43 = arith.constant 0xFF800000 : f32
    %cst_44 = arith.constant 0.000000e+00 : f64
    %cst_45 = arith.constant 0.000000e+00 : f32
    %c30522 = arith.constant 30522 : index
    %c3 = arith.constant 3 : index
    %c0_i64 = arith.constant 0 : i64
    %cst_46 = arith.constant dense_resource<torch_tensor_30522_768_torch.float32> : tensor<30522x768xf32>
    %0 = bufferization.to_memref %arg2 : memref<1x128xi64>
    %1 = bufferization.to_memref %arg1 : memref<1x128xi64>
    %2 = bufferization.to_memref %arg1 : memref<1x128xi64>
    %3 = bufferization.to_memref %cst_31 : memref<768xf32>
    %4 = bufferization.to_memref %cst_30 : memref<768xf32>
    %5 = bufferization.to_memref %cst_29 : memref<768x768xf32>
    %6 = bufferization.to_memref %cst_28 : memref<768xf32>
    %7 = bufferization.to_memref %cst_27 : memref<768x768xf32>
    %8 = bufferization.to_memref %cst_26 : memref<768xf32>
    %9 = bufferization.to_memref %cst_25 : memref<768x768xf32>
    %10 = bufferization.to_memref %cst_24 : memref<768xf32>
    %11 = bufferization.to_memref %cst_23 : memref<768x768xf32>
    %12 = bufferization.to_memref %cst_22 : memref<768xf32>
    %13 = bufferization.to_memref %cst_21 : memref<768xf32>
    %14 = bufferization.to_memref %cst_20 : memref<768xf32>
    %15 = bufferization.to_memref %cst_19 : memref<3072x768xf32>
    %16 = bufferization.to_memref %cst_18 : memref<3072xf32>
    %17 = bufferization.to_memref %cst_17 : memref<768x3072xf32>
    %18 = bufferization.to_memref %cst_16 : memref<768xf32>
    %19 = bufferization.to_memref %cst_15 : memref<768xf32>
    %20 = bufferization.to_memref %cst_14 : memref<768xf32>
    %21 = bufferization.to_memref %cst_13 : memref<768x768xf32>
    %22 = bufferization.to_memref %cst_12 : memref<768xf32>
    %23 = bufferization.to_memref %cst_11 : memref<768x768xf32>
    %24 = bufferization.to_memref %cst_10 : memref<768xf32>
    %25 = bufferization.to_memref %cst_9 : memref<768x768xf32>
    %26 = bufferization.to_memref %cst_8 : memref<768xf32>
    %27 = bufferization.to_memref %cst_7 : memref<768x768xf32>
    %28 = bufferization.to_memref %cst_6 : memref<768xf32>
    %29 = bufferization.to_memref %cst_5 : memref<768xf32>
    %30 = bufferization.to_memref %cst_4 : memref<768xf32>
    %31 = bufferization.to_memref %cst_3 : memref<3072x768xf32>
    %32 = bufferization.to_memref %cst_2 : memref<3072xf32>
    %33 = bufferization.to_memref %cst_1 : memref<768x3072xf32>
    %34 = bufferization.to_memref %cst_0 : memref<768xf32>
    %35 = bufferization.to_memref %cst : memref<f64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128xi1>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        %88 = affine.load %2[%c0, %arg4] : memref<1x128xi64>
        %89 = arith.cmpi sgt, %88, %c0_i64 : i64
        affine.store %89, %alloc[%arg3, %arg4] : memref<1x128xi1>
      }
    }
    %36 = bufferization.to_tensor %alloc : memref<1x128xi1>
    %expanded = tensor.expand_shape %36 [[0, 1], [2, 3, 4, 5]] : tensor<1x128xi1> into tensor<1x1x1x1x1x128xi1>
    %37 = bufferization.to_memref %expanded : memref<1x1x1x1x1x128xi1>
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<1x1x128x1x1x128xi1>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 1 {
              affine.for %arg8 = 0 to 128 {
                %88 = affine.load %37[%arg3, %arg4, %c0, %arg6, %arg7, %arg8] : memref<1x1x1x1x1x128xi1>
                affine.store %88, %alloc_47[%arg3, %arg4, %arg5, %arg6, %arg7, %arg8] : memref<1x1x128x1x1x128xi1>
              }
            }
          }
        }
      }
    }
    %38 = bufferization.to_tensor %alloc_47 : memref<1x1x128x1x1x128xi1>
    %collapsed = tensor.collapse_shape %38 [[0], [1, 2], [3, 4, 5]] : tensor<1x1x128x1x1x128xi1> into tensor<1x128x128xi1>
    %expanded_48 = tensor.expand_shape %collapsed [[0], [1, 2], [3]] : tensor<1x128x128xi1> into tensor<1x1x128x128xi1>
    %39 = bufferization.to_memref %expanded_48 : memref<1x1x128x128xi1>
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %1[%arg3, %arg4] : memref<1x128xi64>
          %89 = arith.index_cast %88 : i64 to index
          %90 = arith.cmpi slt, %89, %c30522 : index
          cf.assert %90, "index must be smaller than dim size"
          %91 = arith.cmpi sge, %88, %c0_i64 : i64
          cf.assert %91, "index must be larger or equal to 0"
          %extracted = tensor.extract %cst_46[%89, %arg5] : tensor<30522x768xf32>
          affine.store %extracted, %alloc_49[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 128, 768] [1, 1, 1] : tensor<1x512x768xf32> to tensor<1x128x768xf32>
    %40 = bufferization.to_memref %extracted_slice : memref<1x128x768xf32>
    %alloc_50 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_49[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %40[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_50[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_51 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %0[%arg3, %arg4] : memref<1x128xi64>
          %89 = arith.index_cast %88 : i64 to index
          %90 = arith.cmpi slt, %89, %c3 : index
          cf.assert %90, "index must be smaller than dim size"
          %91 = arith.cmpi sge, %88, %c0_i64 : i64
          cf.assert %91, "index must be larger or equal to 0"
          %extracted = tensor.extract %cst_32[%89, %arg5] : tensor<3x768xf32>
          affine.store %extracted, %alloc_51[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_50[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_51[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_52[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          affine.store %cst_45, %alloc_53[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_53, %alloc_54 : memref<1x128x1xf32> to memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_52[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_54[%arg3, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_54[%arg3, %arg4, %c0] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_55 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_54[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.divf %88, %cst_38 : f32
          affine.store %89, %alloc_55[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_56 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_52[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = arith.extf %88 : f32 to f64
          affine.store %89, %alloc_56[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_57 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          affine.store %cst_44, %alloc_57[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_58 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_58 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_56[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_58[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_58[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_58[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_37 : f64
          affine.store %89, %alloc_59[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_56[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_59[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.subf %88, %89 : f64
          affine.store %90, %alloc_60[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_61 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_60[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_60[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %90 = arith.mulf %88, %89 : f64
          affine.store %90, %alloc_61[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_62 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_62 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_61[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_62[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_62[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_63 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_62[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_42 : f64
          affine.store %89, %alloc_63[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_64 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_63[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.truncf %88 : f64 to f32
          affine.store %89, %alloc_64[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_65 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_64[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = math.sqrt %88 : f32
          affine.store %89, %alloc_65[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_66 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_52[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_55[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.subf %88, %89 : f32
          affine.store %90, %alloc_66[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_67 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %3[%arg5] : memref<768xf32>
          %89 = affine.load %alloc_66[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_67[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_68 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_65[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.truncf %cst_39 : f64 to f32
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_68[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_69 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_67[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_68[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.divf %88, %89 : f32
          affine.store %90, %alloc_69[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_69[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %4[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_70[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %5[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_71[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_70[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          affine.store %88, %alloc_72[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_73 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_71[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_73[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_74 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.store %cst_45, %alloc_74[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_75 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_75 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_72[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_73[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_75[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_75[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_76 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_75[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %6[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_76[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %41 = bufferization.to_tensor %alloc_76 : memref<1x128x768xf32>
    %expanded_77 = tensor.expand_shape %41 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %42 = bufferization.to_memref %expanded_77 : memref<1x128x12x64xf32>
    %alloc_78 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %42[%arg3, %arg4, %arg5, %arg6] : memref<1x128x12x64xf32>
            affine.store %88, %alloc_78[%arg3, %arg5, %arg4, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %7[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_79[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_80 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_79[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_80[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_81 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_72[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_80[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_81[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_81[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_82 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_81[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %8[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_82[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %43 = bufferization.to_tensor %alloc_82 : memref<1x128x768xf32>
    %expanded_83 = tensor.expand_shape %43 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %44 = bufferization.to_memref %expanded_83 : memref<1x128x12x64xf32>
    %alloc_84 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %9[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_84[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_85 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_84[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_85[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_86 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_86 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_72[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_85[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_86[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_86[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_86[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %10[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_87[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %45 = bufferization.to_tensor %alloc_87 : memref<1x128x768xf32>
    %expanded_88 = tensor.expand_shape %45 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %46 = bufferization.to_memref %expanded_88 : memref<1x128x12x64xf32>
    %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %46[%arg3, %arg4, %arg5, %arg6] : memref<1x128x12x64xf32>
            affine.store %88, %alloc_89[%arg3, %arg5, %arg4, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_90 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %44[%arg3, %arg4, %arg5, %arg6] : memref<1x128x12x64xf32>
            affine.store %88, %alloc_90[%arg3, %arg5, %arg6, %arg4] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %alloc_78[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
            affine.store %88, %alloc_91[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %47 = bufferization.to_tensor %alloc_91 : memref<1x12x128x64xf32>
    %alloc_92 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_90[%c0, %arg4, %arg5, %arg6] : memref<1x12x64x128xf32>
            affine.store %88, %alloc_92[%arg3, %arg4, %arg5, %arg6] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %48 = bufferization.to_tensor %alloc_92 : memref<1x12x64x128xf32>
    %collapsed_93 = tensor.collapse_shape %47 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %49 = bufferization.to_memref %collapsed_93 : memref<12x128x64xf32>
    %collapsed_94 = tensor.collapse_shape %48 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %50 = bufferization.to_memref %collapsed_94 : memref<12x64x128xf32>
    %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<12x128x128xf32>
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 128 {
          affine.store %cst_45, %alloc_95[%arg3, %arg4, %arg5] : memref<12x128x128xf32>
        }
      }
    }
    %alloc_96 = memref.alloc() {alignment = 64 : i64} : memref<12x128x128xf32>
    memref.copy %alloc_95, %alloc_96 : memref<12x128x128xf32> to memref<12x128x128xf32>
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %49[%arg3, %arg4, %arg6] : memref<12x128x64xf32>
            %89 = affine.load %50[%arg3, %arg6, %arg5] : memref<12x64x128xf32>
            %90 = affine.load %alloc_96[%arg3, %arg4, %arg5] : memref<12x128x128xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_96[%arg3, %arg4, %arg5] : memref<12x128x128xf32>
          }
        }
      }
    }
    %51 = bufferization.to_tensor %alloc_96 : memref<12x128x128xf32>
    %expanded_97 = tensor.expand_shape %51 [[0, 1], [2], [3]] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %52 = bufferization.to_memref %expanded_97 : memref<1x12x128x128xf32>
    %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %52[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = arith.divf %88, %cst_36 : f32
            affine.store %89, %alloc_98[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_99 = memref.alloc() {alignment = 64 : i64} : memref<1x1x128x128xi1>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %39[%c0, %c0, %arg5, %arg6] : memref<1x1x128x128xi1>
            %89 = arith.extui %88 : i1 to i64
            %90 = arith.cmpi eq, %89, %c0_i64 : i64
            affine.store %90, %alloc_99[%arg3, %arg4, %arg5, %arg6] : memref<1x1x128x128xi1>
          }
        }
      }
    }
    %alloc_100 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %53 = affine.load %35[] : memref<f64>
    %54 = arith.truncf %53 : f64 to f32
    affine.store %54, %alloc_100[] : memref<f32>
    %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_99[%c0, %c0, %arg5, %arg6] : memref<1x1x128x128xi1>
            %89 = affine.load %alloc_100[] : memref<f32>
            %90 = affine.load %alloc_98[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %91 = arith.select %88, %89, %90 : f32
            affine.store %91, %alloc_101[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_102 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128xi64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.store %c0_i64, %alloc_102[%arg3, %arg4, %arg5] : memref<1x12x128xi64>
        }
      }
    }
    %alloc_103 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.store %cst_43, %alloc_103[%arg3, %arg4, %arg5] : memref<1x12x128xf32>
        }
      }
    }
    %alloc_104 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128xf32>
    memref.copy %alloc_103, %alloc_104 : memref<1x12x128xf32> to memref<1x12x128xf32>
    %alloc_105 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128xi64>
    memref.copy %alloc_102, %alloc_105 : memref<1x12x128xi64> to memref<1x12x128xi64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_101[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %alloc_104[%arg3, %arg4, %arg5] : memref<1x12x128xf32>
            %90 = affine.load %alloc_105[%arg3, %arg4, %arg5] : memref<1x12x128xi64>
            %91 = arith.index_cast %arg6 : index to i64
            %92 = arith.maximumf %88, %89 : f32
            %93 = arith.cmpf ogt, %88, %89 : f32
            %94 = arith.select %93, %91, %90 : i64
            affine.store %92, %alloc_104[%arg3, %arg4, %arg5] : memref<1x12x128xf32>
            affine.store %94, %alloc_105[%arg3, %arg4, %arg5] : memref<1x12x128xi64>
          }
        }
      }
    }
    %55 = bufferization.to_tensor %alloc_104 : memref<1x12x128xf32>
    %expanded_106 = tensor.expand_shape %55 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %56 = bufferization.to_memref %expanded_106 : memref<1x12x128x1xf32>
    %alloc_107 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_101[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %56[%c0, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
            %90 = arith.subf %88, %89 : f32
            affine.store %90, %alloc_107[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_108 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_107[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = math.exp %88 : f32
            affine.store %89, %alloc_108[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 1 {
            affine.store %cst_45, %alloc_109[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x1xf32>
          }
        }
      }
    }
    %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x1xf32>
    memref.copy %alloc_109, %alloc_110 : memref<1x12x128x1xf32> to memref<1x12x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_108[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %alloc_110[%arg3, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
            %90 = arith.addf %88, %89 : f32
            affine.store %90, %alloc_110[%arg3, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
          }
        }
      }
    }
    %alloc_111 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_108[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %alloc_110[%c0, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
            %90 = arith.divf %88, %89 : f32
            affine.store %90, %alloc_111[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_112 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_111[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            affine.store %88, %alloc_112[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %57 = bufferization.to_tensor %alloc_112 : memref<1x12x128x128xf32>
    %alloc_113 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %alloc_89[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
            affine.store %88, %alloc_113[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %58 = bufferization.to_tensor %alloc_113 : memref<1x12x128x64xf32>
    %collapsed_114 = tensor.collapse_shape %57 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %59 = bufferization.to_memref %collapsed_114 : memref<12x128x128xf32>
    %collapsed_115 = tensor.collapse_shape %58 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %60 = bufferization.to_memref %collapsed_115 : memref<12x128x64xf32>
    %alloc_116 = memref.alloc() {alignment = 64 : i64} : memref<12x128x64xf32>
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 64 {
          affine.store %cst_45, %alloc_116[%arg3, %arg4, %arg5] : memref<12x128x64xf32>
        }
      }
    }
    %alloc_117 = memref.alloc() {alignment = 64 : i64} : memref<12x128x64xf32>
    memref.copy %alloc_116, %alloc_117 : memref<12x128x64xf32> to memref<12x128x64xf32>
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %59[%arg3, %arg4, %arg6] : memref<12x128x128xf32>
            %89 = affine.load %60[%arg3, %arg6, %arg5] : memref<12x128x64xf32>
            %90 = affine.load %alloc_117[%arg3, %arg4, %arg5] : memref<12x128x64xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_117[%arg3, %arg4, %arg5] : memref<12x128x64xf32>
          }
        }
      }
    }
    %61 = bufferization.to_tensor %alloc_117 : memref<12x128x64xf32>
    %expanded_118 = tensor.expand_shape %61 [[0, 1], [2], [3]] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %62 = bufferization.to_memref %expanded_118 : memref<1x12x128x64xf32>
    %alloc_119 = memref.alloc() {alignment = 64 : i64} : memref<1x128x12x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %62[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
            affine.store %88, %alloc_119[%arg3, %arg5, %arg4, %arg6] : memref<1x128x12x64xf32>
          }
        }
      }
    }
    %63 = bufferization.to_tensor %alloc_119 : memref<1x128x12x64xf32>
    %collapsed_120 = tensor.collapse_shape %63 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %64 = bufferization.to_memref %collapsed_120 : memref<1x128x768xf32>
    %alloc_121 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %11[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_121[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_122 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %64[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          affine.store %88, %alloc_122[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_123 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_121[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_123[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_124 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_124 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_122[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_123[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_124[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_124[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_125 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_124[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %12[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_125[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_126 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_52[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_125[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_126[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_127 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_53, %alloc_127 : memref<1x128x1xf32> to memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_126[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_127[%arg3, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_127[%arg3, %arg4, %c0] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_128 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_127[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.divf %88, %cst_38 : f32
          affine.store %89, %alloc_128[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_129 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_126[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = arith.extf %88 : f32 to f64
          affine.store %89, %alloc_129[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_130 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_130 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_129[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_130[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_130[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_131 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_130[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_37 : f64
          affine.store %89, %alloc_131[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_132 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_129[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_131[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.subf %88, %89 : f64
          affine.store %90, %alloc_132[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_133 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_132[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_132[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %90 = arith.mulf %88, %89 : f64
          affine.store %90, %alloc_133[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_134 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_134 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_133[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_134[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_134[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_135 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_134[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_42 : f64
          affine.store %89, %alloc_135[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_136 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_135[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.truncf %88 : f64 to f32
          affine.store %89, %alloc_136[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_137 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_136[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = math.sqrt %88 : f32
          affine.store %89, %alloc_137[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_138 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_126[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_128[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.subf %88, %89 : f32
          affine.store %90, %alloc_138[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_139 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %13[%arg5] : memref<768xf32>
          %89 = affine.load %alloc_138[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_139[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_140 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_137[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.truncf %cst_39 : f64 to f32
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_140[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_141 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_139[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_140[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.divf %88, %89 : f32
          affine.store %90, %alloc_141[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_142 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_141[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %14[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_142[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_143 = memref.alloc() {alignment = 64 : i64} : memref<768x3072xf32>
    affine.for %arg3 = 0 to 3072 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %15[%arg3, %arg4] : memref<3072x768xf32>
        affine.store %88, %alloc_143[%arg4, %arg3] : memref<768x3072xf32>
      }
    }
    %alloc_144 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_142[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          affine.store %88, %alloc_144[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_145 = memref.alloc() {alignment = 64 : i64} : memref<1x768x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_143[%arg4, %arg5] : memref<768x3072xf32>
          affine.store %88, %alloc_145[%arg3, %arg4, %arg5] : memref<1x768x3072xf32>
        }
      }
    }
    %alloc_146 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          affine.store %cst_45, %alloc_146[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_147 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    memref.copy %alloc_146, %alloc_147 : memref<1x128x3072xf32> to memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_144[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_145[%arg3, %arg6, %arg5] : memref<1x768x3072xf32>
            %90 = affine.load %alloc_147[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_147[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
          }
        }
      }
    }
    %alloc_148 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_147[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = affine.load %16[%arg5] : memref<3072xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_148[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_149 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_148[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.mulf %88, %cst_35 : f32
          affine.store %89, %alloc_149[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_150 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_148[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = math.powf %88, %cst_34 : f32
          affine.store %89, %alloc_150[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_151 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_150[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.truncf %cst_40 : f64 to f32
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_151[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_152 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_148[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = affine.load %alloc_151[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_152[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_153 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_152[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.truncf %cst_41 : f64 to f32
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_153[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_154 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_153[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = math.tanh %88 : f32
          affine.store %89, %alloc_154[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_155 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_154[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.addf %88, %cst_33 : f32
          affine.store %89, %alloc_155[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_156 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_149[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = affine.load %alloc_155[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_156[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_157 = memref.alloc() {alignment = 64 : i64} : memref<3072x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 3072 {
        %88 = affine.load %17[%arg3, %arg4] : memref<768x3072xf32>
        affine.store %88, %alloc_157[%arg4, %arg3] : memref<3072x768xf32>
      }
    }
    %alloc_158 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_156[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          affine.store %88, %alloc_158[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_159 = memref.alloc() {alignment = 64 : i64} : memref<1x3072x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 3072 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_157[%arg4, %arg5] : memref<3072x768xf32>
          affine.store %88, %alloc_159[%arg3, %arg4, %arg5] : memref<1x3072x768xf32>
        }
      }
    }
    %alloc_160 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_160 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 3072 {
            %88 = affine.load %alloc_158[%arg3, %arg4, %arg6] : memref<1x128x3072xf32>
            %89 = affine.load %alloc_159[%arg3, %arg6, %arg5] : memref<1x3072x768xf32>
            %90 = affine.load %alloc_160[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_160[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_161 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_160[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %18[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_161[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_162 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_126[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_161[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_162[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_163 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_53, %alloc_163 : memref<1x128x1xf32> to memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_162[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_163[%arg3, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_163[%arg3, %arg4, %c0] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_164 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_163[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.divf %88, %cst_38 : f32
          affine.store %89, %alloc_164[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_165 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_162[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = arith.extf %88 : f32 to f64
          affine.store %89, %alloc_165[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_166 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_166 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_165[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_166[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_166[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_167 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_166[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_37 : f64
          affine.store %89, %alloc_167[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_168 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_165[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_167[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.subf %88, %89 : f64
          affine.store %90, %alloc_168[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_169 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_168[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_168[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %90 = arith.mulf %88, %89 : f64
          affine.store %90, %alloc_169[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_170 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_170 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_169[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_170[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_170[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_171 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_170[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_42 : f64
          affine.store %89, %alloc_171[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_172 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_171[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.truncf %88 : f64 to f32
          affine.store %89, %alloc_172[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_173 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_172[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = math.sqrt %88 : f32
          affine.store %89, %alloc_173[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_174 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_162[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_164[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.subf %88, %89 : f32
          affine.store %90, %alloc_174[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_175 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %19[%arg5] : memref<768xf32>
          %89 = affine.load %alloc_174[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_175[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_176 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_173[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.truncf %cst_39 : f64 to f32
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_176[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_177 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_175[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_176[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.divf %88, %89 : f32
          affine.store %90, %alloc_177[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_178 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_177[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %20[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_178[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_179 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %21[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_179[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_180 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_178[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          affine.store %88, %alloc_180[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_181 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_179[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_181[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_182 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_182 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_180[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_181[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_182[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_182[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_183 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_182[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %22[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_183[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %65 = bufferization.to_tensor %alloc_183 : memref<1x128x768xf32>
    %expanded_184 = tensor.expand_shape %65 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %66 = bufferization.to_memref %expanded_184 : memref<1x128x12x64xf32>
    %alloc_185 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %66[%arg3, %arg4, %arg5, %arg6] : memref<1x128x12x64xf32>
            affine.store %88, %alloc_185[%arg3, %arg5, %arg4, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_186 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %23[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_186[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_187 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_186[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_187[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_188 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_188 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_180[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_187[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_188[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_188[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_189 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_188[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %24[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_189[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %67 = bufferization.to_tensor %alloc_189 : memref<1x128x768xf32>
    %expanded_190 = tensor.expand_shape %67 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %68 = bufferization.to_memref %expanded_190 : memref<1x128x12x64xf32>
    %alloc_191 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %25[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_191[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_192 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_191[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_192[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_193 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_193 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_180[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_192[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_193[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_193[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_194 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_193[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %26[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_194[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %69 = bufferization.to_tensor %alloc_194 : memref<1x128x768xf32>
    %expanded_195 = tensor.expand_shape %69 [[0], [1], [2, 3]] : tensor<1x128x768xf32> into tensor<1x128x12x64xf32>
    %70 = bufferization.to_memref %expanded_195 : memref<1x128x12x64xf32>
    %alloc_196 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %70[%arg3, %arg4, %arg5, %arg6] : memref<1x128x12x64xf32>
            affine.store %88, %alloc_196[%arg3, %arg5, %arg4, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_197 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 12 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %68[%arg3, %arg4, %arg5, %arg6] : memref<1x128x12x64xf32>
            affine.store %88, %alloc_197[%arg3, %arg5, %arg6, %arg4] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %alloc_198 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %alloc_185[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
            affine.store %88, %alloc_198[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %71 = bufferization.to_tensor %alloc_198 : memref<1x12x128x64xf32>
    %alloc_199 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_197[%c0, %arg4, %arg5, %arg6] : memref<1x12x64x128xf32>
            affine.store %88, %alloc_199[%arg3, %arg4, %arg5, %arg6] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %72 = bufferization.to_tensor %alloc_199 : memref<1x12x64x128xf32>
    %collapsed_200 = tensor.collapse_shape %71 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %73 = bufferization.to_memref %collapsed_200 : memref<12x128x64xf32>
    %collapsed_201 = tensor.collapse_shape %72 [[0, 1], [2], [3]] : tensor<1x12x64x128xf32> into tensor<12x64x128xf32>
    %74 = bufferization.to_memref %collapsed_201 : memref<12x64x128xf32>
    %alloc_202 = memref.alloc() {alignment = 64 : i64} : memref<12x128x128xf32>
    memref.copy %alloc_95, %alloc_202 : memref<12x128x128xf32> to memref<12x128x128xf32>
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %73[%arg3, %arg4, %arg6] : memref<12x128x64xf32>
            %89 = affine.load %74[%arg3, %arg6, %arg5] : memref<12x64x128xf32>
            %90 = affine.load %alloc_202[%arg3, %arg4, %arg5] : memref<12x128x128xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_202[%arg3, %arg4, %arg5] : memref<12x128x128xf32>
          }
        }
      }
    }
    %75 = bufferization.to_tensor %alloc_202 : memref<12x128x128xf32>
    %expanded_203 = tensor.expand_shape %75 [[0, 1], [2], [3]] : tensor<12x128x128xf32> into tensor<1x12x128x128xf32>
    %76 = bufferization.to_memref %expanded_203 : memref<1x12x128x128xf32>
    %alloc_204 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %76[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = arith.divf %88, %cst_36 : f32
            affine.store %89, %alloc_204[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_205 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_99[%c0, %c0, %arg5, %arg6] : memref<1x1x128x128xi1>
            %89 = affine.load %alloc_100[] : memref<f32>
            %90 = affine.load %alloc_204[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %91 = arith.select %88, %89, %90 : f32
            affine.store %91, %alloc_205[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_206 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128xf32>
    memref.copy %alloc_103, %alloc_206 : memref<1x12x128xf32> to memref<1x12x128xf32>
    %alloc_207 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128xi64>
    memref.copy %alloc_102, %alloc_207 : memref<1x12x128xi64> to memref<1x12x128xi64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_205[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %alloc_206[%arg3, %arg4, %arg5] : memref<1x12x128xf32>
            %90 = affine.load %alloc_207[%arg3, %arg4, %arg5] : memref<1x12x128xi64>
            %91 = arith.index_cast %arg6 : index to i64
            %92 = arith.maximumf %88, %89 : f32
            %93 = arith.cmpf ogt, %88, %89 : f32
            %94 = arith.select %93, %91, %90 : i64
            affine.store %92, %alloc_206[%arg3, %arg4, %arg5] : memref<1x12x128xf32>
            affine.store %94, %alloc_207[%arg3, %arg4, %arg5] : memref<1x12x128xi64>
          }
        }
      }
    }
    %77 = bufferization.to_tensor %alloc_206 : memref<1x12x128xf32>
    %expanded_208 = tensor.expand_shape %77 [[0], [1], [2, 3]] : tensor<1x12x128xf32> into tensor<1x12x128x1xf32>
    %78 = bufferization.to_memref %expanded_208 : memref<1x12x128x1xf32>
    %alloc_209 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_205[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %78[%c0, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
            %90 = arith.subf %88, %89 : f32
            affine.store %90, %alloc_209[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_210 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_209[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = math.exp %88 : f32
            affine.store %89, %alloc_210[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_211 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x1xf32>
    memref.copy %alloc_109, %alloc_211 : memref<1x12x128x1xf32> to memref<1x12x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_210[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %alloc_211[%arg3, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
            %90 = arith.addf %88, %89 : f32
            affine.store %90, %alloc_211[%arg3, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
          }
        }
      }
    }
    %alloc_212 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_210[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            %89 = affine.load %alloc_211[%c0, %arg4, %arg5, %c0] : memref<1x12x128x1xf32>
            %90 = arith.divf %88, %89 : f32
            affine.store %90, %alloc_212[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_213 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x128xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %alloc_212[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
            affine.store %88, %alloc_213[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %79 = bufferization.to_tensor %alloc_213 : memref<1x12x128x128xf32>
    %alloc_214 = memref.alloc() {alignment = 64 : i64} : memref<1x12x128x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %alloc_196[%c0, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
            affine.store %88, %alloc_214[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %80 = bufferization.to_tensor %alloc_214 : memref<1x12x128x64xf32>
    %collapsed_215 = tensor.collapse_shape %79 [[0, 1], [2], [3]] : tensor<1x12x128x128xf32> into tensor<12x128x128xf32>
    %81 = bufferization.to_memref %collapsed_215 : memref<12x128x128xf32>
    %collapsed_216 = tensor.collapse_shape %80 [[0, 1], [2], [3]] : tensor<1x12x128x64xf32> into tensor<12x128x64xf32>
    %82 = bufferization.to_memref %collapsed_216 : memref<12x128x64xf32>
    %alloc_217 = memref.alloc() {alignment = 64 : i64} : memref<12x128x64xf32>
    memref.copy %alloc_116, %alloc_217 : memref<12x128x64xf32> to memref<12x128x64xf32>
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 128 {
            %88 = affine.load %81[%arg3, %arg4, %arg6] : memref<12x128x128xf32>
            %89 = affine.load %82[%arg3, %arg6, %arg5] : memref<12x128x64xf32>
            %90 = affine.load %alloc_217[%arg3, %arg4, %arg5] : memref<12x128x64xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_217[%arg3, %arg4, %arg5] : memref<12x128x64xf32>
          }
        }
      }
    }
    %83 = bufferization.to_tensor %alloc_217 : memref<12x128x64xf32>
    %expanded_218 = tensor.expand_shape %83 [[0, 1], [2], [3]] : tensor<12x128x64xf32> into tensor<1x12x128x64xf32>
    %84 = bufferization.to_memref %expanded_218 : memref<1x12x128x64xf32>
    %alloc_219 = memref.alloc() {alignment = 64 : i64} : memref<1x128x12x64xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 12 {
        affine.for %arg5 = 0 to 128 {
          affine.for %arg6 = 0 to 64 {
            %88 = affine.load %84[%arg3, %arg4, %arg5, %arg6] : memref<1x12x128x64xf32>
            affine.store %88, %alloc_219[%arg3, %arg5, %arg4, %arg6] : memref<1x128x12x64xf32>
          }
        }
      }
    }
    %85 = bufferization.to_tensor %alloc_219 : memref<1x128x12x64xf32>
    %collapsed_220 = tensor.collapse_shape %85 [[0], [1], [2, 3]] : tensor<1x128x12x64xf32> into tensor<1x128x768xf32>
    %86 = bufferization.to_memref %collapsed_220 : memref<1x128x768xf32>
    %alloc_221 = memref.alloc() {alignment = 64 : i64} : memref<768x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %27[%arg3, %arg4] : memref<768x768xf32>
        affine.store %88, %alloc_221[%arg4, %arg3] : memref<768x768xf32>
      }
    }
    %alloc_222 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %86[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          affine.store %88, %alloc_222[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_223 = memref.alloc() {alignment = 64 : i64} : memref<1x768x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_221[%arg4, %arg5] : memref<768x768xf32>
          affine.store %88, %alloc_223[%arg3, %arg4, %arg5] : memref<1x768x768xf32>
        }
      }
    }
    %alloc_224 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_224 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_222[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_223[%arg3, %arg6, %arg5] : memref<1x768x768xf32>
            %90 = affine.load %alloc_224[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_224[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_225 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_224[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %28[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_225[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_226 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_162[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_225[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_226[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_227 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    memref.copy %alloc_53, %alloc_227 : memref<1x128x1xf32> to memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_226[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_227[%arg3, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_227[%arg3, %arg4, %c0] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_228 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_227[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.divf %88, %cst_38 : f32
          affine.store %89, %alloc_228[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_229 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_226[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = arith.extf %88 : f32 to f64
          affine.store %89, %alloc_229[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_230 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_230 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_229[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_230[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_230[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_231 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_230[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_37 : f64
          affine.store %89, %alloc_231[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_232 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_229[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_231[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.subf %88, %89 : f64
          affine.store %90, %alloc_232[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_233 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_232[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_232[%c0, %arg4, %arg5] : memref<1x128x768xf64>
          %90 = arith.mulf %88, %89 : f64
          affine.store %90, %alloc_233[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
        }
      }
    }
    %alloc_234 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    memref.copy %alloc_57, %alloc_234 : memref<1x128x1xf64> to memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_233[%arg3, %arg4, %arg5] : memref<1x128x768xf64>
          %89 = affine.load %alloc_234[%arg3, %arg4, %c0] : memref<1x128x1xf64>
          %90 = arith.addf %88, %89 : f64
          affine.store %90, %alloc_234[%arg3, %arg4, %c0] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_235 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf64>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_234[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.divf %88, %cst_42 : f64
          affine.store %89, %alloc_235[%arg3, %arg4, %arg5] : memref<1x128x1xf64>
        }
      }
    }
    %alloc_236 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_235[%c0, %arg4, %c0] : memref<1x128x1xf64>
          %89 = arith.truncf %88 : f64 to f32
          affine.store %89, %alloc_236[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_237 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_236[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = math.sqrt %88 : f32
          affine.store %89, %alloc_237[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_238 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_226[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_228[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.subf %88, %89 : f32
          affine.store %90, %alloc_238[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_239 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %29[%arg5] : memref<768xf32>
          %89 = affine.load %alloc_238[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_239[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_240 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 1 {
          %88 = affine.load %alloc_237[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %89 = arith.truncf %cst_39 : f64 to f32
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_240[%arg3, %arg4, %arg5] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_241 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_239[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_240[%c0, %arg4, %c0] : memref<1x128x1xf32>
          %90 = arith.divf %88, %89 : f32
          affine.store %90, %alloc_241[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_242 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_241[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %30[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_242[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_243 = memref.alloc() {alignment = 64 : i64} : memref<768x3072xf32>
    affine.for %arg3 = 0 to 3072 {
      affine.for %arg4 = 0 to 768 {
        %88 = affine.load %31[%arg3, %arg4] : memref<3072x768xf32>
        affine.store %88, %alloc_243[%arg4, %arg3] : memref<768x3072xf32>
      }
    }
    %alloc_244 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_242[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          affine.store %88, %alloc_244[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_245 = memref.alloc() {alignment = 64 : i64} : memref<1x768x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_243[%arg4, %arg5] : memref<768x3072xf32>
          affine.store %88, %alloc_245[%arg3, %arg4, %arg5] : memref<1x768x3072xf32>
        }
      }
    }
    %alloc_246 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    memref.copy %alloc_146, %alloc_246 : memref<1x128x3072xf32> to memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          affine.for %arg6 = 0 to 768 {
            %88 = affine.load %alloc_244[%arg3, %arg4, %arg6] : memref<1x128x768xf32>
            %89 = affine.load %alloc_245[%arg3, %arg6, %arg5] : memref<1x768x3072xf32>
            %90 = affine.load %alloc_246[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_246[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
          }
        }
      }
    }
    %alloc_247 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_246[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = affine.load %32[%arg5] : memref<3072xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_247[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_248 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_247[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.mulf %88, %cst_35 : f32
          affine.store %89, %alloc_248[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_249 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_247[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = math.powf %88, %cst_34 : f32
          affine.store %89, %alloc_249[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_250 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_249[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.truncf %cst_40 : f64 to f32
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_250[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_251 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_247[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = affine.load %alloc_250[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_251[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_252 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_251[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.truncf %cst_41 : f64 to f32
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_252[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_253 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_252[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = math.tanh %88 : f32
          affine.store %89, %alloc_253[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_254 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_253[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = arith.addf %88, %cst_33 : f32
          affine.store %89, %alloc_254[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_255 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_248[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %89 = affine.load %alloc_254[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          %90 = arith.mulf %88, %89 : f32
          affine.store %90, %alloc_255[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_256 = memref.alloc() {alignment = 64 : i64} : memref<3072x768xf32>
    affine.for %arg3 = 0 to 768 {
      affine.for %arg4 = 0 to 3072 {
        %88 = affine.load %33[%arg3, %arg4] : memref<768x3072xf32>
        affine.store %88, %alloc_256[%arg4, %arg3] : memref<3072x768xf32>
      }
    }
    %alloc_257 = memref.alloc() {alignment = 64 : i64} : memref<1x128x3072xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 3072 {
          %88 = affine.load %alloc_255[%c0, %arg4, %arg5] : memref<1x128x3072xf32>
          affine.store %88, %alloc_257[%arg3, %arg4, %arg5] : memref<1x128x3072xf32>
        }
      }
    }
    %alloc_258 = memref.alloc() {alignment = 64 : i64} : memref<1x3072x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 3072 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_256[%arg4, %arg5] : memref<3072x768xf32>
          affine.store %88, %alloc_258[%arg3, %arg4, %arg5] : memref<1x3072x768xf32>
        }
      }
    }
    %alloc_259 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    memref.copy %alloc_74, %alloc_259 : memref<1x128x768xf32> to memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          affine.for %arg6 = 0 to 3072 {
            %88 = affine.load %alloc_257[%arg3, %arg4, %arg6] : memref<1x128x3072xf32>
            %89 = affine.load %alloc_258[%arg3, %arg6, %arg5] : memref<1x3072x768xf32>
            %90 = affine.load %alloc_259[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
            %91 = arith.mulf %88, %89 : f32
            %92 = arith.addf %90, %91 : f32
            affine.store %92, %alloc_259[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
          }
        }
      }
    }
    %alloc_260 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_259[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %34[%arg5] : memref<768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_260[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_261 = memref.alloc() {alignment = 64 : i64} : memref<1x128x768xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 768 {
          %88 = affine.load %alloc_226[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %89 = affine.load %alloc_260[%c0, %arg4, %arg5] : memref<1x128x768xf32>
          %90 = arith.addf %88, %89 : f32
          affine.store %90, %alloc_261[%arg3, %arg4, %arg5] : memref<1x128x768xf32>
        }
      }
    }
    %87 = bufferization.to_tensor %alloc_261 : memref<1x128x768xf32>
    return %87 : tensor<1x128x768xf32>
  }
}

