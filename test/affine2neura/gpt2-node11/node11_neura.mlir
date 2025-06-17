module {
  memref.global @input : memref<1x16x64xf32> = uninitialized
  memref.global @output : memref<1x16xf32> = uninitialized
  func.func @_Z6node11v() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @output : memref<1x16xf32>
    %1 = memref.get_global @input : memref<1x16x64xf32>
    %2 = neura.constant {value = 0 : index} : index
    %3 = neura.constant {value = 16 : index} : index
    %4 = neura.constant {value = 1 : index} : index
    neura.br %2 : index to ^bb2
  ^bb1:  // pred: ^bb8
    return %c0_i32 : i32
  ^bb2(%5: index):  // 2 preds: ^bb0, ^bb4
    neura.loop_control current_index : %5, step : %4, bound : %3, loop_type : "lt" then ^bb3(%5 : index) else ^bb8
  ^bb3(%6: index):  // pred: ^bb2
    %7 = neura.constant {value = 0 : index} : index
    %8 = neura.constant {value = 64 : index} : index
    %9 = neura.constant {value = 1 : index} : index
    neura.br %7 : index to ^bb5
  ^bb4:  // pred: ^bb7
    neura.br %6 : index to ^bb2
  ^bb5(%10: index):  // 2 preds: ^bb3, ^bb6
    neura.loop_control current_index : %10, step : %9, bound : %8, loop_type : "lt" then ^bb6(%10 : index) else ^bb7
  ^bb6(%11: index):  // pred: ^bb5
    %12 = neura.constant {value = 0 : index} : index
    %13 = neura.load_indexed memref<1x16x64xf32> %1[%12, %6, %11] : f32
    %14 = neura.constant {value = 0 : index} : index
    %15 = neura.load_indexed memref<1x16xf32> %0[%14, %6] : f32
    %16 = arith.addf %15, %13 : f32
    %17 = neura.constant {value = 0 : index} : index
    neura.store_indexed %16 to memref<1x16xf32> %0[%17, %6] : f32
    neura.br %11 : index to ^bb5
  ^bb7:  // pred: ^bb5
    neura.br :  to ^bb4
  ^bb8:  // pred: ^bb2
    neura.br :  to ^bb1
  }
}

