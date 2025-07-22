// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

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

// CHECK:   func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 32 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 128 : i32}> : () -> i32
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = -128 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 32 : i32}> : () -> i32
// CHECK-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
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

// CTRL2DATA:   func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<memref<?x32x32xi32>, i1>) -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.grant_once"(%0) : (!neura.data<memref<?x32x32xi32>, i1>) -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.grant_always"(%3) : (!neura.data<memref<?x32xi32>, i1>) -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%3) : (!neura.data<memref<?x32xi32>, i1>) -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %8 = "neura.grant_once"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 32 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %10 = "neura.grant_always"(%9) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%9) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_always"(%12) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %14 = "neura.grant_once"(%12) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %15 = "neura.constant"() <{predicate = true, value = -128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %16 = "neura.grant_always"(%15) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%15) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %18 = "neura.constant"() <{predicate = true, value = 32 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = "neura.grant_always"(%18) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = "neura.grant_once"(%18) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = "neura.grant_always"(%21) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.grant_once"(%21) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %25 = "neura.grant_always"(%24) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %26 = "neura.grant_once"(%24) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %27 = "neura.cast"(%24) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %28 = "neura.grant_once"(%27) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %30 = "neura.phi"(%29, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %31 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = "neura.phi"(%31, %28) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = "neura.cast"(%32) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %34 = "neura.icmp"(%33, %30) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %35 = "neura.not"(%34) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %36 = neura.grant_predicate %26, %34 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %37 = "neura.cast"(%36) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %38 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %39 = "neura.phi"(%38, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %40 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %41 = "neura.phi"(%40, %37) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = "neura.cast"(%41) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %43 = "neura.icmp"(%42, %39) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %23, %43 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %45 = neura.grant_predicate %5, %43 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %33, %43 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %42, %43 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %26, %43 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %49 = "neura.not"(%43) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %26, %49 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %51 = neura.grant_predicate %23, %49 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %44 to %45[%46, %47 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %52 = "neura.cast"(%48) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %53 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %54 = "neura.phi"(%53, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %55 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = "neura.phi"(%55, %52) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %57 = "neura.cast"(%56) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %58 = "neura.icmp"(%57, %54) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %59 = neura.grant_predicate %2, %58 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %60 = neura.grant_predicate %33, %58 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %61 = neura.grant_predicate %42, %58 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %62 = neura.grant_predicate %57, %58 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %63 = neura.grant_predicate %5, %58 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %64 = neura.grant_predicate %8, %58 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %65 = neura.grant_predicate %11, %58 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %66 = "neura.not"(%58) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %67 = neura.grant_predicate %42, %66 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %68 = neura.grant_predicate %8, %66 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %69 = neura.grant_predicate %11, %66 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %70 = neura.load_indexed %59[%60, %61, %62 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %71 = neura.load_indexed %63[%60, %61 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %72 = "neura.add"(%71, %70) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %72 to %63[%60, %61 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %73 = "neura.add"(%62, %64) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %74 = "neura.cast"(%73) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %74 -> %55 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %65 -> %53 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %75 = "neura.add"(%67, %68) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %76 = "neura.cast"(%75) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %76 -> %40 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %69 -> %38 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %77 = "neura.cast"(%50) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %78 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %79 = "neura.phi"(%78, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %80 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %81 = "neura.phi"(%80, %51) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %82 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %83 = "neura.phi"(%82, %77) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %84 = "neura.cast"(%83) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %85 = "neura.icmp"(%84, %79) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %86 = neura.grant_predicate %5, %85 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %87 = neura.grant_predicate %33, %85 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %88 = neura.grant_predicate %84, %85 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %89 = neura.grant_predicate %81, %85 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %90 = neura.grant_predicate %8, %85 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %91 = neura.grant_predicate %11, %85 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %92 = "neura.not"(%85) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %93 = neura.grant_predicate %81, %92 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %94 = neura.grant_predicate %20, %92 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %95 = neura.grant_predicate %26, %92 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %96 = neura.load_indexed %86[%87, %88 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %97 = "neura.add"(%89, %96) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %98 = "neura.add"(%88, %90) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %99 = "neura.cast"(%98) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %99 -> %82 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %97 -> %80 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %91 -> %78 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %100 = "neura.div"(%93, %94) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %101 = "neura.cast"(%95) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %102 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %103 = "neura.phi"(%102, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %104 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %105 = "neura.phi"(%104, %101) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %106 = "neura.cast"(%105) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %107 = "neura.icmp"(%106, %103) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %108 = neura.grant_predicate %26, %107 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %109 = neura.grant_predicate %17, %107 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %110 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %111 = neura.grant_predicate %26, %110 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %112 = "neura.cast"(%108) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %113 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %114 = "neura.phi"(%113, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %115 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %116 = "neura.phi"(%115, %109) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %117 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %118 = "neura.phi"(%117, %112) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %119 = "neura.cast"(%118) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %120 = "neura.icmp"(%119, %114) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %121 = neura.grant_predicate %2, %120 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %122 = neura.grant_predicate %119, %120 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %123 = neura.grant_predicate %106, %120 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %124 = neura.grant_predicate %33, %120 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %125 = neura.grant_predicate %116, %120 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %126 = neura.grant_predicate %8, %120 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %127 = neura.grant_predicate %11, %120 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %128 = "neura.not"(%120) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %129 = neura.grant_predicate %5, %128 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %130 = neura.grant_predicate %33, %128 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %131 = neura.grant_predicate %106, %128 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %132 = neura.grant_predicate %116, %128 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %133 = neura.grant_predicate %14, %128 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %134 = neura.grant_predicate %8, %128 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %135 = neura.grant_predicate %11, %128 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %136 = neura.load_indexed %121[%122, %123, %124 : !neura.data<index, i1>, !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %137 = "neura.icmp"(%136, %125) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %138 = "neura.sel"(%136, %125, %137) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %139 = "neura.add"(%122, %126) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %140 = "neura.cast"(%139) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %140 -> %117 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %138 -> %115 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %127 -> %113 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %141 = neura.load_indexed %129[%130, %131 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %142 = "neura.mul"(%141, %132) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %143 = "neura.div"(%142, %133) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %143 to %129[%130, %131 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %144 = "neura.add"(%131, %134) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %145 = "neura.cast"(%144) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %145 -> %104 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %135 -> %102 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %146 = "neura.cast"(%111) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %147 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %148 = "neura.phi"(%147, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %149 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %150 = "neura.phi"(%149, %146) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %151 = "neura.cast"(%150) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %152 = "neura.icmp"(%151, %148) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %153 = neura.grant_predicate %5, %152 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %154 = neura.grant_predicate %33, %152 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %155 = neura.grant_predicate %151, %152 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %156 = neura.grant_predicate %100, %152 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %157 = "neura.not"(%152) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %158 = neura.grant_predicate %33, %157 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %159 = neura.grant_predicate %8, %157 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %160 = neura.grant_predicate %11, %157 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %161 = neura.load_indexed %153[%154, %155 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %162 = "neura.icmp"(%161, %156) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %163 = neura.grant_predicate %100, %162 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %164 = neura.grant_predicate %5, %162 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %165 = neura.grant_predicate %33, %162 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %166 = neura.grant_predicate %151, %162 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %167 = "neura.not"(%162) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %168 = neura.grant_predicate %151, %167 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %169 = neura.grant_predicate %8, %167 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %170 = neura.grant_predicate %11, %167 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %163 to %164[%165, %166 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %171 = "neura.phi"(%170, %11) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %172 = "neura.phi"(%169, %8) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %173 = "neura.phi"(%168, %151) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %174 = "neura.add"(%173, %172) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %175 = "neura.cast"(%174) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %175 -> %149 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %171 -> %147 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %176 = "neura.add"(%158, %159) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %177 = "neura.cast"(%176) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %177 -> %31 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %160 -> %29 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }