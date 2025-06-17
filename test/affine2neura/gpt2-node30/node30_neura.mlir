module {
  memref.global @A : memref<1x4x16x64xf32> = uninitialized
  memref.global @C : memref<1x4x16x64xf32> = uninitialized
  func.func @_Z6node30v() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+01 : f32
    %0 = llvm.mlir.undef : i32
    %1 = memref.get_global @C : memref<1x4x16x64xf32>
    %2 = memref.get_global @A : memref<1x4x16x64xf32>
    %3 = neura.constant {value = 0 : index} : index
    %4 = neura.constant {value = 4 : index} : index
    %5 = neura.constant {value = 1 : index} : index
    neura.br %3 : index to ^bb2
  ^bb1:  // pred: ^bb12
    return %0 : i32
  ^bb2(%6: index):  // 2 preds: ^bb0, ^bb4
    neura.loop_control current_index : %6, step : %5, bound : %4, loop_type : "lt" then ^bb3(%6 : index) else ^bb12
  ^bb3(%7: index):  // pred: ^bb2
    %8 = neura.constant {value = 0 : index} : index
    %9 = neura.constant {value = 16 : index} : index
    %10 = neura.constant {value = 1 : index} : index
    neura.br %8 : index to ^bb5
  ^bb4:  // pred: ^bb11
    neura.br %7 : index to ^bb2
  ^bb5(%11: index):  // 2 preds: ^bb3, ^bb7
    neura.loop_control current_index : %11, step : %10, bound : %9, loop_type : "lt" then ^bb6(%11 : index) else ^bb11
  ^bb6(%12: index):  // pred: ^bb5
    %13 = neura.constant {value = 0 : index} : index
    %14 = neura.constant {value = 64 : index} : index
    %15 = neura.constant {value = 1 : index} : index
    neura.br %13 : index to ^bb8
  ^bb7:  // pred: ^bb10
    neura.br %12 : index to ^bb5
  ^bb8(%16: index):  // 2 preds: ^bb6, ^bb9
    neura.loop_control current_index : %16, step : %15, bound : %14, loop_type : "lt" then ^bb9(%16 : index) else ^bb10
  ^bb9(%17: index):  // pred: ^bb8
    %18 = neura.constant {value = 0 : index} : index
    %19 = neura.load_indexed memref<1x4x16x64xf32> %2[%18, %7, %12, %17] : f32
    %20 = arith.mulf %19, %cst : f32
    %21 = neura.constant {value = 0 : index} : index
    neura.store_indexed %20 to memref<1x4x16x64xf32> %1[%21, %7, %12, %17] : f32
    neura.br %17 : index to ^bb8
  ^bb10:  // pred: ^bb8
    neura.br :  to ^bb7
  ^bb11:  // pred: ^bb5
    neura.br :  to ^bb4
  ^bb12:  // pred: ^bb2
    neura.br :  to ^bb1
  }
}

