// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

module attributes {} {
  func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128_i32 = arith.constant 128 : i32
    %c-128_i32 = arith.constant -128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 32 {
        affine.store %c0_i32, %arg1[%arg2, %arg3] : memref<?x32xi32>
        affine.for %arg4 = 0 to 32 {
          %2 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<?x32x32xi32>
          %3 = affine.load %arg1[%arg2, %arg3] : memref<?x32xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg1[%arg2, %arg3] : memref<?x32xi32>
        }
      }
      %0 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %c0_i32) -> (i32) {
        %2 = affine.load %arg1[%arg2, %arg3] : memref<?x32xi32>
        %3 = arith.addi %arg4, %2 : i32
        affine.yield %3 : i32
      }
      %1 = arith.divsi %0, %c32_i32 : i32
      affine.for %arg3 = 0 to 32 {
        %2 = affine.for %arg4 = 0 to 32 iter_args(%arg5 = %c-128_i32) -> (i32) {
          %6 = affine.load %arg0[%arg4, %arg3, %arg2] : memref<?x32x32xi32>
          %7 = arith.cmpi sgt, %6, %arg5 : i32
          %8 = arith.select %7, %6, %arg5 : i32
          affine.yield %8 : i32
        }
        %3 = affine.load %arg1[%arg2, %arg3] : memref<?x32xi32>
        %4 = arith.muli %3, %2 : i32
        %5 = arith.divsi %4, %c128_i32 : i32
        affine.store %5, %arg1[%arg2, %arg3] : memref<?x32xi32>
      }
      affine.for %arg3 = 0 to 32 {
        %2 = affine.load %arg1[%arg2, %arg3] : memref<?x32xi32>
        %3 = arith.cmpi sgt, %2, %1 : i32
        scf.if %3 {
          affine.store %1, %arg1[%arg2, %arg3] : memref<?x32xi32>
        }
      }
    }
    return
  }
}

// CHECK: func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 32 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 128 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{value = -128 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 32 : i32}> : () -> i32
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %6 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %7 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %7 : i64 to ^bb1
// CHECK-NEXT:   ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb22
// CHECK-NEXT:     %9 = "neura.cast"(%8) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %10 = "neura.icmp"(%9, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %10 : i1 then to ^bb2 else to ^bb23
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %11 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %11 : i64 to ^bb3
// CHECK-NEXT:   ^bb3(%12: i64):  // 2 preds: ^bb2, ^bb7
// CHECK-NEXT:     %13 = "neura.cast"(%12) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %14 = "neura.icmp"(%13, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %14 : i1 then to ^bb4 else to ^bb8
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     neura.store_indexed %5 to %arg1[%9, %13 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %15 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %15 : i64 to ^bb5
// CHECK-NEXT:   ^bb5(%16: i64):  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT:     %17 = "neura.cast"(%16) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %18 = "neura.icmp"(%17, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %18 : i1 then to ^bb6 else to ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %19 = neura.load_indexed %arg0[%9, %13, %17 : index, index, index] memref<?x32x32xi32> : i32
// CHECK-NEXT:     %20 = neura.load_indexed %arg1[%9, %13 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %21 = "neura.add"(%20, %19) : (i32, i32) -> i32
// CHECK-NEXT:     neura.store_indexed %21 to %arg1[%9, %13 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %22 = "neura.add"(%17, %0) : (index, index) -> index
// CHECK-NEXT:     %23 = "neura.cast"(%22) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %23 : i64 to ^bb5
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     %24 = "neura.add"(%13, %0) : (index, index) -> index
// CHECK-NEXT:     %25 = "neura.cast"(%24) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %25 : i64 to ^bb3
// CHECK-NEXT:   ^bb8:  // pred: ^bb3
// CHECK-NEXT:     %26 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %26, %5 : i64, i32 to ^bb9
// CHECK-NEXT:   ^bb9(%27: i64, %28: i32):  // 2 preds: ^bb8, ^bb10
// CHECK-NEXT:     %29 = "neura.cast"(%27) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %30 = "neura.icmp"(%29, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %30 : i1 then to ^bb10 else to ^bb11
// CHECK-NEXT:   ^bb10:  // pred: ^bb9
// CHECK-NEXT:     %31 = neura.load_indexed %arg1[%9, %29 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %32 = "neura.add"(%28, %31) : (i32, i32) -> i32
// CHECK-NEXT:     %33 = "neura.add"(%29, %0) : (index, index) -> index
// CHECK-NEXT:     %34 = "neura.cast"(%33) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %34, %32 : i64, i32 to ^bb9
// CHECK-NEXT:   ^bb11:  // pred: ^bb9
// CHECK-NEXT:     %35 = "neura.div"(%28, %4) : (i32, i32) -> i32
// CHECK-NEXT:     %36 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %36 : i64 to ^bb12
// CHECK-NEXT:   ^bb12(%37: i64):  // 2 preds: ^bb11, ^bb16
// CHECK-NEXT:     %38 = "neura.cast"(%37) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %39 = "neura.icmp"(%38, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %39 : i1 then to ^bb13 else to ^bb17
// CHECK-NEXT:   ^bb13:  // pred: ^bb12
// CHECK-NEXT:     %40 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %40, %3 : i64, i32 to ^bb14
// CHECK-NEXT:   ^bb14(%41: i64, %42: i32):  // 2 preds: ^bb13, ^bb15
// CHECK-NEXT:     %43 = "neura.cast"(%41) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %44 = "neura.icmp"(%43, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %44 : i1 then to ^bb15 else to ^bb16
// CHECK-NEXT:   ^bb15:  // pred: ^bb14
// CHECK-NEXT:     %45 = neura.load_indexed %arg0[%43, %38, %9 : index, index, index] memref<?x32x32xi32> : i32
// CHECK-NEXT:     %46 = "neura.icmp"(%45, %42) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     %47 = "neura.sel"(%45, %42, %46) : (i32, i32, i1) -> i32
// CHECK-NEXT:     %48 = "neura.add"(%43, %0) : (index, index) -> index
// CHECK-NEXT:     %49 = "neura.cast"(%48) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %49, %47 : i64, i32 to ^bb14
// CHECK-NEXT:   ^bb16:  // pred: ^bb14
// CHECK-NEXT:     %50 = neura.load_indexed %arg1[%9, %38 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %51 = "neura.mul"(%50, %42) : (i32, i32) -> i32
// CHECK-NEXT:     %52 = "neura.div"(%51, %2) : (i32, i32) -> i32
// CHECK-NEXT:     neura.store_indexed %52 to %arg1[%9, %38 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %53 = "neura.add"(%38, %0) : (index, index) -> index
// CHECK-NEXT:     %54 = "neura.cast"(%53) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %54 : i64 to ^bb12
// CHECK-NEXT:   ^bb17:  // pred: ^bb12
// CHECK-NEXT:     %55 = "neura.cast"(%6) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %55 : i64 to ^bb18
// CHECK-NEXT:   ^bb18(%56: i64):  // 2 preds: ^bb17, ^bb21
// CHECK-NEXT:     %57 = "neura.cast"(%56) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %58 = "neura.icmp"(%57, %1) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %58 : i1 then to ^bb19 else to ^bb22
// CHECK-NEXT:   ^bb19:  // pred: ^bb18
// CHECK-NEXT:     %59 = neura.load_indexed %arg1[%9, %57 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     %60 = "neura.icmp"(%59, %35) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %60 : i1 then to ^bb20 else to ^bb21
// CHECK-NEXT:   ^bb20:  // pred: ^bb19
// CHECK-NEXT:     neura.store_indexed %35 to %arg1[%9, %57 : index, index] memref<?x32xi32> : i32
// CHECK-NEXT:     neura.br to ^bb21
// CHECK-NEXT:   ^bb21:  // 2 preds: ^bb19, ^bb20
// CHECK-NEXT:     %61 = "neura.add"(%57, %0) : (index, index) -> index
// CHECK-NEXT:     %62 = "neura.cast"(%61) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %62 : i64 to ^bb18
// CHECK-NEXT:   ^bb22:  // pred: ^bb18
// CHECK-NEXT:     %63 = "neura.add"(%9, %0) : (index, index) -> index
// CHECK-NEXT:     %64 = "neura.cast"(%63) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %64 : i64 to ^bb1
// CHECK-NEXT:   ^bb23:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }

// CTRL2DATA: func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 32 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_always"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_always"(%4) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = -128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.grant_once"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.constant"() <{value = 32 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.grant_always"(%9) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.grant_always"(%11) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%11) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %14 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %15 = "neura.grant_always"(%14) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %16 = "neura.cast"(%14) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %19 = "neura.phi"(%18, %17) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %20 = "neura.cast"(%19) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %21 = "neura.icmp"(%20, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %22 = "neura.not"(%21) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %23 = neura.grant_predicate %15, %21 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %24 = "neura.cast"(%23) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = "neura.phi"(%25, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %27 = "neura.cast"(%26) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %28 = "neura.icmp"(%27, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %29 = "neura.not"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %12, %28 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %31 = neura.grant_predicate %20, %28 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %27, %28 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %30 to %arg1[%31, %32 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %15, %28 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %34 = "neura.cast"(%33) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = "neura.phi"(%35, %34) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %37 = "neura.cast"(%36) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %38 = "neura.icmp"(%37, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %39 = "neura.not"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %20, %38 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %41 = neura.grant_predicate %27, %38 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %37, %38 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %43 = neura.load_indexed %arg0[%40, %41, %42 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %44 = neura.load_indexed %arg1[%40, %41 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %45 = "neura.add"(%44, %43) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %45 to %arg1[%40, %41 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %1, %38 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %47 = "neura.add"(%42, %46) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %48 = "neura.cast"(%47) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %48 -> %35 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = neura.grant_predicate %27, %39 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %1, %39 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %51 = "neura.add"(%49, %50) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %52 = "neura.cast"(%51) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %52 -> %25 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %53 = neura.grant_predicate %15, %29 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %54 = "neura.cast"(%53) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %13, %29 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %56 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %57 = "neura.phi"(%56, %55) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %58 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = "neura.phi"(%58, %54) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %60 = "neura.cast"(%59) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %61 = "neura.icmp"(%60, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %62 = "neura.not"(%61) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %63 = neura.grant_predicate %20, %61 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %64 = neura.grant_predicate %60, %61 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %65 = neura.load_indexed %arg1[%63, %64 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %66 = "neura.add"(%57, %65) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %67 = neura.grant_predicate %1, %61 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %68 = "neura.add"(%64, %67) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %69 = "neura.cast"(%68) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %69 -> %58 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %66 -> %56 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %70 = neura.grant_predicate %10, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %71 = "neura.div"(%57, %70) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %72 = neura.grant_predicate %15, %62 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %73 = "neura.cast"(%72) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %74 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %75 = "neura.phi"(%74, %73) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %76 = "neura.cast"(%75) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %77 = "neura.icmp"(%76, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %78 = "neura.not"(%77) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %79 = neura.grant_predicate %15, %77 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %80 = "neura.cast"(%79) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %81 = neura.grant_predicate %8, %77 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %82 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %83 = "neura.phi"(%82, %81) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %84 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %85 = "neura.phi"(%84, %80) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %86 = "neura.cast"(%85) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %87 = "neura.icmp"(%86, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %88 = "neura.not"(%87) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %89 = neura.grant_predicate %86, %87 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %90 = neura.grant_predicate %76, %87 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %91 = neura.grant_predicate %20, %87 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %92 = neura.load_indexed %arg0[%89, %90, %91 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %93 = "neura.icmp"(%92, %83) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %94 = "neura.sel"(%92, %83, %93) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %95 = neura.grant_predicate %1, %87 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %96 = "neura.add"(%89, %95) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %97 = "neura.cast"(%96) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %97 -> %84 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %94 -> %82 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %98 = neura.grant_predicate %20, %88 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %99 = neura.grant_predicate %76, %88 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %100 = neura.load_indexed %arg1[%98, %99 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %101 = "neura.mul"(%100, %83) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %102 = neura.grant_predicate %5, %88 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %103 = "neura.div"(%101, %102) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %103 to %arg1[%98, %99 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %104 = neura.grant_predicate %1, %88 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %105 = "neura.add"(%99, %104) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %106 = "neura.cast"(%105) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %106 -> %74 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %107 = neura.grant_predicate %15, %78 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %108 = "neura.cast"(%107) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %109 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %110 = "neura.phi"(%109, %108) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %111 = "neura.cast"(%110) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %112 = "neura.icmp"(%111, %3) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %113 = "neura.not"(%112) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %114 = neura.grant_predicate %20, %112 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %115 = neura.grant_predicate %111, %112 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %116 = neura.load_indexed %arg1[%114, %115 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %117 = neura.grant_predicate %71, %112 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %118 = "neura.icmp"(%116, %117) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %119 = "neura.not"(%118) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %120 = neura.grant_predicate %71, %118 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %121 = neura.grant_predicate %20, %118 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %122 = neura.grant_predicate %111, %118 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %120 to %arg1[%121, %122 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x32xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %123 = neura.grant_predicate %111, %119 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %124 = neura.grant_predicate %1, %119 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %125 = "neura.add"(%123, %124) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %126 = "neura.cast"(%125) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %126 -> %109 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %127 = neura.grant_predicate %20, %113 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %128 = neura.grant_predicate %1, %113 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %129 = "neura.add"(%127, %128) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %130 = "neura.cast"(%129) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %130 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }