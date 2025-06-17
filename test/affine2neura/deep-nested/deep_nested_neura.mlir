module {
  memref.global @input_data : memref<3x3x3xi32> = uninitialized
  memref.global @output_data : memref<3x3x3xi32> = uninitialized
  func.func @_Z11deep_nestedv() -> i32 attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @output_data : memref<3x3x3xi32>
    %1 = memref.get_global @input_data : memref<3x3x3xi32>
    %2 = neura.constant {value = 0 : index} : index
    %3 = neura.constant {value = 3 : index} : index
    %4 = neura.constant {value = 1 : index} : index
    neura.br %2 : index to ^bb2
  ^bb1:  // pred: ^bb40
    return %c0_i32 : i32
  ^bb2(%5: index):  // 2 preds: ^bb0, ^bb4
    neura.loop_control current_index : %5, step : %4, bound : %3, loop_type : "lt" then ^bb3(%5 : index) else ^bb40
  ^bb3(%6: index):  // pred: ^bb2
    %7 = neura.constant {value = 0 : index} : index
    %8 = neura.constant {value = 3 : index} : index
    %9 = neura.constant {value = 1 : index} : index
    neura.br %7 : index to ^bb5
  ^bb4:  // pred: ^bb39
    neura.br %6 : index to ^bb2
  ^bb5(%10: index):  // 2 preds: ^bb3, ^bb7
    neura.loop_control current_index : %10, step : %9, bound : %8, loop_type : "lt" then ^bb6(%10 : index) else ^bb39
  ^bb6(%11: index):  // pred: ^bb5
    %12 = neura.constant {value = 0 : index} : index
    %13 = neura.constant {value = 3 : index} : index
    %14 = neura.constant {value = 1 : index} : index
    neura.br %12 : index to ^bb8
  ^bb7:  // pred: ^bb38
    neura.br %11 : index to ^bb5
  ^bb8(%15: index):  // 2 preds: ^bb6, ^bb10
    neura.loop_control current_index : %15, step : %14, bound : %13, loop_type : "lt" then ^bb9(%15 : index) else ^bb38
  ^bb9(%16: index):  // pred: ^bb8
    %17 = neura.constant {value = 0 : index} : index
    %18 = neura.constant {value = 3 : index} : index
    %19 = neura.constant {value = 1 : index} : index
    neura.br %17 : index to ^bb11
  ^bb10:  // pred: ^bb37
    neura.br %16 : index to ^bb8
  ^bb11(%20: index):  // 2 preds: ^bb9, ^bb13
    neura.loop_control current_index : %20, step : %19, bound : %18, loop_type : "lt" then ^bb12(%20 : index) else ^bb37
  ^bb12(%21: index):  // pred: ^bb11
    %22 = neura.constant {value = 0 : index} : index
    %23 = neura.constant {value = 3 : index} : index
    %24 = neura.constant {value = 1 : index} : index
    neura.br %22 : index to ^bb14
  ^bb13:  // pred: ^bb36
    neura.br %21 : index to ^bb11
  ^bb14(%25: index):  // 2 preds: ^bb12, ^bb16
    neura.loop_control current_index : %25, step : %24, bound : %23, loop_type : "lt" then ^bb15(%25 : index) else ^bb36
  ^bb15(%26: index):  // pred: ^bb14
    %27 = neura.constant {value = 0 : index} : index
    %28 = neura.constant {value = 3 : index} : index
    %29 = neura.constant {value = 1 : index} : index
    neura.br %27 : index to ^bb17
  ^bb16:  // pred: ^bb35
    neura.br %26 : index to ^bb14
  ^bb17(%30: index):  // 2 preds: ^bb15, ^bb19
    neura.loop_control current_index : %30, step : %29, bound : %28, loop_type : "lt" then ^bb18(%30 : index) else ^bb35
  ^bb18(%31: index):  // pred: ^bb17
    %32 = neura.constant {value = 0 : index} : index
    %33 = neura.constant {value = 3 : index} : index
    %34 = neura.constant {value = 1 : index} : index
    neura.br %32 : index to ^bb20
  ^bb19:  // pred: ^bb34
    neura.br %31 : index to ^bb17
  ^bb20(%35: index):  // 2 preds: ^bb18, ^bb22
    neura.loop_control current_index : %35, step : %34, bound : %33, loop_type : "lt" then ^bb21(%35 : index) else ^bb34
  ^bb21(%36: index):  // pred: ^bb20
    %37 = neura.constant {value = 0 : index} : index
    %38 = neura.constant {value = 3 : index} : index
    %39 = neura.constant {value = 1 : index} : index
    neura.br %37 : index to ^bb23
  ^bb22:  // pred: ^bb33
    neura.br %36 : index to ^bb20
  ^bb23(%40: index):  // 2 preds: ^bb21, ^bb25
    neura.loop_control current_index : %40, step : %39, bound : %38, loop_type : "lt" then ^bb24(%40 : index) else ^bb33
  ^bb24(%41: index):  // pred: ^bb23
    %42 = neura.load_indexed memref<3x3x3xi32> %1[%6, %11, %16] : i32
    %43 = neura.constant {value = 0 : index} : index
    %44 = neura.constant {value = 3 : index} : index
    %45 = neura.constant {value = 1 : index} : index
    neura.br %43 : index to ^bb26
  ^bb25:  // pred: ^bb32
    neura.br %41 : index to ^bb23
  ^bb26(%46: index):  // 2 preds: ^bb24, ^bb28
    neura.loop_control current_index : %46, step : %45, bound : %44, loop_type : "lt" then ^bb27(%46 : index) else ^bb32
  ^bb27(%47: index):  // pred: ^bb26
    %48 = neura.constant {value = 0 : index} : index
    %49 = neura.constant {value = 3 : index} : index
    %50 = neura.constant {value = 1 : index} : index
    neura.br %48 : index to ^bb29
  ^bb28:  // pred: ^bb31
    neura.br %47 : index to ^bb26
  ^bb29(%51: index):  // 2 preds: ^bb27, ^bb30
    neura.loop_control current_index : %51, step : %50, bound : %49, loop_type : "lt" then ^bb30(%51 : index) else ^bb31
  ^bb30(%52: index):  // pred: ^bb29
    %53 = neura.load_indexed memref<3x3x3xi32> %0[%6, %11, %16] : i32
    %54 = arith.addi %53, %42 : i32
    neura.store_indexed %54 to memref<3x3x3xi32> %0[%6, %11, %16] : i32
    neura.br %52 : index to ^bb29
  ^bb31:  // pred: ^bb29
    neura.br :  to ^bb28
  ^bb32:  // pred: ^bb26
    neura.br :  to ^bb25
  ^bb33:  // pred: ^bb23
    neura.br :  to ^bb22
  ^bb34:  // pred: ^bb20
    neura.br :  to ^bb19
  ^bb35:  // pred: ^bb17
    neura.br :  to ^bb16
  ^bb36:  // pred: ^bb14
    neura.br :  to ^bb13
  ^bb37:  // pred: ^bb11
    neura.br :  to ^bb10
  ^bb38:  // pred: ^bb8
    neura.br :  to ^bb7
  ^bb39:  // pred: ^bb5
    neura.br :  to ^bb4
  ^bb40:  // pred: ^bb2
    neura.br :  to ^bb1
  }
}

