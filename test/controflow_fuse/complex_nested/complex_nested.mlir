// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: | FileCheck %s

// RUN: mlir-neura-opt %t-llvm.mlir \
// RUN: --assign-accelerator \
// RUN: --lower-arith-to-neura \
// RUN: --lower-memref-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --canonicalize-cast \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s -check-prefix=CTRL2DATA

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

// CTRL2DATA:      func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:   %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:   %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x32x32xi32>, i1>) -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:   %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x32xi32>, i1>) -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %4 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %6 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %8 = "neura.constant"() <{predicate = true, value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %9 = "neura.grant_once"(%8) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %10 = "neura.constant"() <{predicate = true, value = -128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %11 = "neura.grant_once"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %12 = "neura.constant"() <{predicate = true, value = 32 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %13 = "neura.grant_once"(%12) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %14 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %15 = "neura.grant_once"(%14) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %16 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %18 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %19 = "neura.phi"(%18, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %20 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %21 = "neura.phi"(%20, %17) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %22 = "neura.icmp"(%21, %19) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %23 = neura.grant_predicate %17, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %24 = neura.grant_predicate %7, %22 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %25 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %26 = "neura.phi"(%25, %24) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %27 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %28 = "neura.phi"(%27, %23) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %29 = "neura.icmp"(%28, %26) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %31 = "neura.and"(%22, %30) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %32 = "neura.and"(%22, %29) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %33 = neura.grant_predicate %15, %29 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %34 = neura.grant_predicate %3, %29 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %35 = neura.grant_predicate %21, %29 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %36 = neura.grant_predicate %28, %29 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %37 = neura.grant_predicate %17, %29 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %38 = neura.grant_predicate %17, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %39 = neura.grant_predicate %15, %30 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   neura.store_indexed %33 to %34[%35, %36 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %40 = neura.grant_predicate %7, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %41 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %42 = "neura.phi"(%41, %40) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %43 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %44 = "neura.phi"(%43, %37) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %45 = "neura.icmp"(%44, %42) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %46 = "neura.not"(%45) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %47 = neura.grant_predicate %1, %45 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:   %48 = neura.grant_predicate %21, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %49 = neura.grant_predicate %28, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %50 = neura.grant_predicate %44, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %51 = neura.grant_predicate %3, %45 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %52 = neura.grant_predicate %5, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %53 = neura.grant_predicate %7, %45 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %54 = neura.grant_predicate %28, %46 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %55 = neura.grant_predicate %5, %46 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %56 = neura.grant_predicate %7, %46 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %57 = neura.load_indexed %47[%48, %49, %50 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %58 = neura.load_indexed %51[%48, %49 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %59 = "neura.add"(%58, %57) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   neura.store_indexed %59 to %51[%48, %49 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %60 = "neura.add"(%50, %52) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %60 -> %43 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %53 -> %41 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %61 = "neura.add"(%54, %55) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %61 -> %27 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %56 -> %25 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %62 = neura.grant_predicate %7, %31 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %63 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %64 = "neura.phi"(%63, %62) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %65 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %66 = "neura.phi"(%65, %39) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %67 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %68 = "neura.phi"(%67, %38) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %69 = "neura.icmp"(%68, %64) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %70 = "neura.not"(%69) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %71 = "neura.and"(%31, %70) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %72 = neura.grant_predicate %3, %69 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %73 = neura.grant_predicate %21, %69 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %74 = neura.grant_predicate %68, %69 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %75 = neura.grant_predicate %66, %69 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %76 = neura.grant_predicate %5, %69 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %77 = neura.grant_predicate %7, %69 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %78 = neura.grant_predicate %66, %70 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %79 = neura.grant_predicate %13, %70 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %80 = neura.grant_predicate %17, %70 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %81 = neura.load_indexed %72[%73, %74 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %82 = "neura.add"(%75, %81) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %83 = "neura.add"(%74, %76) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %83 -> %67 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %82 -> %65 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %77 -> %63 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %84 = "neura.div"(%78, %79) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %85 = neura.grant_predicate %7, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %86 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %87 = "neura.phi"(%86, %85) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %88 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %89 = "neura.phi"(%88, %80) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %90 = "neura.icmp"(%89, %87) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %91 = "neura.not"(%90) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %92 = "neura.and"(%71, %91) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %93 = "neura.and"(%71, %90) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %94 = neura.grant_predicate %17, %90 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %95 = neura.grant_predicate %11, %90 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %96 = neura.grant_predicate %17, %91 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %97 = neura.grant_predicate %7, %93 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %98 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %99 = "neura.phi"(%98, %97) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %100 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %101 = "neura.phi"(%100, %95) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %102 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %103 = "neura.phi"(%102, %94) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %104 = "neura.icmp"(%103, %99) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %105 = "neura.not"(%104) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %106 = neura.grant_predicate %1, %104 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:   %107 = neura.grant_predicate %103, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %108 = neura.grant_predicate %89, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %109 = neura.grant_predicate %21, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %110 = neura.grant_predicate %101, %104 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %111 = neura.grant_predicate %5, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %112 = neura.grant_predicate %7, %104 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %113 = neura.grant_predicate %3, %105 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %114 = neura.grant_predicate %21, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %115 = neura.grant_predicate %89, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %116 = neura.grant_predicate %101, %105 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %117 = neura.grant_predicate %9, %105 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %118 = neura.grant_predicate %5, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %119 = neura.grant_predicate %7, %105 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %120 = neura.load_indexed %106[%107, %108, %109 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %121 = "neura.icmp"(%120, %110) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %122 = "neura.sel"(%120, %110, %121) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %123 = "neura.add"(%107, %111) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %123 -> %102 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %122 -> %100 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %112 -> %98 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %124 = neura.load_indexed %113[%114, %115 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %125 = "neura.mul"(%124, %116) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %126 = "neura.div"(%125, %117) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   neura.store_indexed %126 to %113[%114, %115 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %127 = "neura.add"(%115, %118) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %127 -> %88 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %119 -> %86 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %128 = neura.grant_predicate %7, %92 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %129 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %130 = "neura.phi"(%129, %128) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %131 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %132 = "neura.phi"(%131, %96) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %133 = "neura.icmp"(%132, %130) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %134 = "neura.not"(%133) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %135 = "neura.and"(%92, %133) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %136 = neura.grant_predicate %3, %133 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %137 = neura.grant_predicate %21, %133 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %138 = neura.grant_predicate %132, %133 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %139 = neura.grant_predicate %84, %133 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %140 = neura.grant_predicate %21, %134 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %141 = neura.grant_predicate %5, %134 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %142 = neura.grant_predicate %7, %134 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %143 = neura.load_indexed %136[%137, %138 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %144 = "neura.icmp"(%143, %139) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %145 = "neura.and"(%135, %144) : (!neura.data<i1, i1>, !neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %146 = "neura.not"(%144) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:   %147 = neura.grant_predicate %84, %144 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %148 = neura.grant_predicate %3, %144 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:   %149 = neura.grant_predicate %21, %144 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %150 = neura.grant_predicate %132, %144 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %151 = neura.grant_predicate %132, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %152 = neura.grant_predicate %5, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %153 = neura.grant_predicate %7, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.store_indexed %147 to %148[%149, %150 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:   %154 = neura.grant_predicate %5, %145 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %155 = neura.grant_predicate %7, %145 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %156 = "neura.phi"(%153, %155) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %157 = "neura.phi"(%152, %154) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %158 = "neura.phi"(%151, %132) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %159 = "neura.add"(%158, %157) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %159 -> %131 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %156 -> %129 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   %160 = "neura.add"(%140, %141) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %160 -> %20 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   neura.ctrl_mov %142 -> %18 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:   "neura.return"() : () -> ()
// CTRL2DATA-NEXT: }