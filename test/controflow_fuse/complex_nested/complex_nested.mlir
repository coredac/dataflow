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
// RUN: --promote-func-arg-to-const \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: | FileCheck %s --check-prefix=CTRL2DATA

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

// CHECK:      module {
// CHECK-NEXT:   func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
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
// CHECK-NEXT:     %47 = "neura.sel"(%46, %45, %42) : (i1, i32, i32) -> i32
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
// CHECK-NEXT: }

// CTRL2DATA:      module {
// CTRL2DATA-NEXT:   func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x32x32xi32>, i1>) -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x32xi32>, i1>) -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 32 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = -128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{value = 32 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %14 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %15 = "neura.grant_once"(%14) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %16 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = neura.phi_start %9, %18 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = neura.phi_start %11, %20 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = neura.phi_start %13, %22 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %25 = neura.phi_start %5, %24 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %26 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %27 = neura.phi_start %1, %26 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %28 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %29 = neura.phi_start %3, %28 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %30 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %31 = neura.phi_start %15, %30 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %32 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.phi_start %17, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %34 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.phi_start %7, %34 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %37 = neura.phi_start %17, %36 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %38 = "neura.icmp"(%37, %35) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %39 = neura.grant_predicate %33, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %35, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %41 = neura.grant_predicate %31, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %29, %38 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %43 = neura.grant_predicate %37, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %27, %38 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %45 = neura.grant_predicate %25, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %23, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %21, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %19, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %49 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %50 = neura.phi_start %48, %49 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %51 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %52 = neura.phi_start %47, %51 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %53 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %54 = neura.phi_start %46, %53 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %55 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = neura.phi_start %45, %55 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %57 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %58 = neura.phi_start %44, %57 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %59 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %60 = neura.phi_start %39, %59 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %61 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %62 = neura.phi_start %43, %61 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %63 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %64 = neura.phi_start %42, %63 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %65 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %66 = neura.phi_start %41, %65 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %67 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %68 = neura.phi_start %40, %67 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %69 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %70 = neura.phi_start %39, %69 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %71 = "neura.icmp"(%70, %68) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %72 = neura.grant_predicate %66, %71 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %73 = neura.grant_predicate %64, %71 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %74 = neura.grant_predicate %62, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %75 = neura.grant_predicate %70, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %76 = neura.grant_predicate %60, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %77 = neura.grant_predicate %68, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %78 = neura.grant_predicate %58, %71 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %79 = neura.grant_predicate %56, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %80 = neura.grant_predicate %54, %71 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %81 = neura.grant_predicate %52, %71 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %82 = neura.grant_predicate %50, %71 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %83 = "neura.not"(%71) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %84 = neura.grant_predicate %60, %83 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %85 = neura.grant_predicate %66, %83 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %86 = neura.grant_predicate %68, %83 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %87 = neura.grant_predicate %64, %83 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %88 = neura.grant_predicate %62, %83 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %89 = neura.grant_predicate %56, %83 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %90 = neura.grant_predicate %54, %83 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %91 = neura.grant_predicate %52, %83 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %92 = neura.grant_predicate %58, %83 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %93 = neura.grant_predicate %50, %83 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %94 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %95 = neura.phi_start %85, %94 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %96 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %97 = neura.phi_start %93, %96 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %98 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %99 = neura.phi_start %92, %98 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %100 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %101 = neura.phi_start %91, %100 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %102 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %103 = neura.phi_start %84, %102 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %104 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %105 = neura.phi_start %90, %104 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %106 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %107 = neura.phi_start %89, %106 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %108 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %109 = neura.phi_start %88, %108 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %110 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %111 = neura.phi_start %87, %110 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %112 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %113 = neura.phi_start %86, %112 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %114 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %115 = neura.phi_start %85, %114 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %116 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %117 = neura.phi_start %84, %116 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %118 = "neura.icmp"(%117, %113) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %119 = neura.grant_predicate %111, %118 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %120 = neura.grant_predicate %109, %118 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %121 = neura.grant_predicate %117, %118 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %122 = neura.grant_predicate %115, %118 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %123 = neura.grant_predicate %107, %118 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %124 = neura.grant_predicate %113, %118 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %125 = neura.grant_predicate %105, %118 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %126 = neura.grant_predicate %103, %118 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %127 = neura.grant_predicate %101, %118 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %128 = neura.grant_predicate %99, %118 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %129 = neura.grant_predicate %97, %118 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %130 = neura.grant_predicate %95, %118 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %131 = "neura.not"(%118) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %132 = neura.grant_predicate %115, %131 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %133 = neura.grant_predicate %105, %131 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %134 = neura.grant_predicate %103, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %135 = neura.grant_predicate %113, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %136 = neura.grant_predicate %101, %131 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %137 = neura.grant_predicate %99, %131 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %138 = neura.grant_predicate %109, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %139 = neura.grant_predicate %107, %131 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %140 = neura.grant_predicate %111, %131 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %141 = neura.grant_predicate %97, %131 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %142 = neura.grant_predicate %95, %131 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %143 = "neura.div"(%132, %133) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %144 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %145 = neura.phi_start %133, %144 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %146 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %147 = neura.phi_start %142, %146 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %148 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %149 = neura.phi_start %143, %148 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %150 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %151 = neura.phi_start %141, %150 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %152 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %153 = neura.phi_start %140, %152 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %154 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %155 = neura.phi_start %139, %154 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %156 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %157 = neura.phi_start %138, %156 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %158 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %159 = neura.phi_start %137, %158 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %160 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %161 = neura.phi_start %136, %160 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %162 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %163 = neura.phi_start %134, %162 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %164 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %165 = neura.phi_start %135, %164 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %166 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %167 = neura.phi_start %134, %166 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %168 = "neura.icmp"(%167, %165) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %169 = neura.grant_predicate %163, %168 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %170 = neura.grant_predicate %161, %168 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %171 = neura.grant_predicate %165, %168 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %172 = neura.grant_predicate %159, %168 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %173 = neura.grant_predicate %167, %168 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %174 = neura.grant_predicate %157, %168 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %175 = neura.grant_predicate %155, %168 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %176 = neura.grant_predicate %153, %168 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %177 = neura.grant_predicate %151, %168 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %178 = neura.grant_predicate %149, %168 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %179 = neura.grant_predicate %147, %168 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %180 = neura.grant_predicate %145, %168 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %181 = "neura.not"(%168) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %182 = neura.grant_predicate %163, %181 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %183 = neura.grant_predicate %165, %181 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %184 = neura.grant_predicate %153, %181 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %185 = neura.grant_predicate %157, %181 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %186 = neura.grant_predicate %149, %181 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %187 = neura.grant_predicate %155, %181 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %188 = neura.grant_predicate %147, %181 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %189 = neura.grant_predicate %159, %181 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %190 = neura.grant_predicate %145, %181 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %191 = neura.grant_predicate %161, %181 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %192 = neura.grant_predicate %151, %181 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %193 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %194 = neura.phi_start %192, %193 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %195 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %196 = neura.phi_start %191, %195 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %197 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %198 = neura.phi_start %190, %197 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %199 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %200 = neura.phi_start %189, %199 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %201 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %202 = neura.phi_start %188, %201 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %203 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %204 = neura.phi_start %182, %203 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %205 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %206 = neura.phi_start %187, %205 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %207 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %208 = neura.phi_start %186, %207 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %209 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %210 = neura.phi_start %185, %209 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %211 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %212 = neura.phi_start %184, %211 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %213 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %214 = neura.phi_start %183, %213 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %215 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %216 = neura.phi_start %182, %215 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %217 = "neura.icmp"(%216, %214) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %218 = neura.grant_predicate %212, %217 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %219 = neura.grant_predicate %210, %217 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %220 = neura.grant_predicate %216, %217 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %221 = neura.grant_predicate %208, %217 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %222 = neura.grant_predicate %206, %217 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %223 = neura.grant_predicate %214, %217 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %224 = neura.grant_predicate %204, %217 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %225 = neura.grant_predicate %202, %217 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %226 = neura.grant_predicate %200, %217 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %227 = neura.grant_predicate %198, %217 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %228 = neura.grant_predicate %196, %217 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %229 = neura.grant_predicate %194, %217 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %230 = "neura.not"(%217) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %231 = neura.grant_predicate %210, %230 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %232 = neura.grant_predicate %206, %230 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %233 = neura.grant_predicate %214, %230 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %234 = neura.grant_predicate %204, %230 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %235 = neura.grant_predicate %202, %230 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %236 = neura.grant_predicate %212, %230 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %237 = neura.grant_predicate %200, %230 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %238 = neura.grant_predicate %198, %230 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %239 = neura.grant_predicate %196, %230 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %240 = neura.grant_predicate %194, %230 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %241 = "neura.add"(%231, %232) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %241 -> %36 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %233 -> %34 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %234 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %235 -> %30 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %236 -> %28 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %237 -> %26 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %232 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %238 -> %22 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %239 -> %20 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %240 -> %18 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %242 = neura.load_indexed %218[%219, %220 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %243 = "neura.icmp"(%242, %221) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %244 = neura.grant_predicate %221, %243 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %245 = neura.grant_predicate %218, %243 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %246 = neura.grant_predicate %219, %243 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %247 = neura.grant_predicate %220, %243 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %248 = neura.grant_predicate %222, %243 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %249 = neura.grant_predicate %223, %243 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %250 = neura.grant_predicate %224, %243 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %251 = neura.grant_predicate %225, %243 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %252 = neura.grant_predicate %226, %243 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %253 = neura.grant_predicate %227, %243 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %254 = neura.grant_predicate %228, %243 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %255 = neura.grant_predicate %229, %243 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %256 = "neura.not"(%243) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %257 = neura.grant_predicate %220, %256 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %258 = neura.grant_predicate %222, %256 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %259 = neura.grant_predicate %223, %256 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %260 = neura.grant_predicate %218, %256 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %261 = neura.grant_predicate %219, %256 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %262 = neura.grant_predicate %221, %256 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %263 = neura.grant_predicate %224, %256 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %264 = neura.grant_predicate %225, %256 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %265 = neura.grant_predicate %226, %256 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %266 = neura.grant_predicate %227, %256 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %267 = neura.grant_predicate %228, %256 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %268 = neura.grant_predicate %229, %256 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %244 to %245[%246, %247 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %269 = "neura.phi"(%268, %255) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %270 = "neura.phi"(%267, %254) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %271 = "neura.phi"(%266, %253) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %272 = "neura.phi"(%265, %252) : (!neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1>) -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %273 = "neura.phi"(%264, %251) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %274 = "neura.phi"(%263, %250) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %275 = "neura.phi"(%262, %244) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %276 = "neura.phi"(%261, %246) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %277 = "neura.phi"(%260, %245) : (!neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1>) -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %278 = "neura.phi"(%259, %249) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %279 = "neura.phi"(%258, %248) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %280 = "neura.phi"(%257, %247) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %281 = "neura.add"(%280, %279) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %281 -> %215 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %278 -> %213 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %277 -> %211 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %276 -> %209 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %275 -> %207 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %279 -> %205 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %274 -> %203 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %273 -> %201 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %272 -> %199 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %271 -> %197 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %270 -> %195 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %269 -> %193 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %282 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %283 = neura.phi_start %180, %282 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %284 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %285 = neura.phi_start %179, %284 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %286 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %287 = neura.phi_start %178, %286 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %288 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %289 = neura.phi_start %170, %288 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %290 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %291 = neura.phi_start %169, %290 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %292 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %293 = neura.phi_start %177, %292 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %294 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %295 = neura.phi_start %176, %294 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %296 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %297 = neura.phi_start %175, %296 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %298 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %299 = neura.phi_start %174, %298 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %300 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %301 = neura.phi_start %173, %300 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %302 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %303 = neura.phi_start %172, %302 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %304 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %305 = neura.phi_start %171, %304 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %306 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %307 = neura.phi_start %170, %306 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %308 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %309 = neura.phi_start %169, %308 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %310 = "neura.icmp"(%309, %305) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %311 = neura.grant_predicate %303, %310 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %312 = neura.grant_predicate %309, %310 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %313 = neura.grant_predicate %301, %310 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %314 = neura.grant_predicate %299, %310 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %315 = neura.grant_predicate %307, %310 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %316 = neura.grant_predicate %297, %310 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %317 = neura.grant_predicate %305, %310 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %318 = neura.grant_predicate %295, %310 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %319 = neura.grant_predicate %293, %310 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %320 = neura.grant_predicate %291, %310 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %321 = neura.grant_predicate %289, %310 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %322 = neura.grant_predicate %287, %310 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %323 = neura.grant_predicate %285, %310 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %324 = neura.grant_predicate %283, %310 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %325 = "neura.not"(%310) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %326 = neura.grant_predicate %295, %325 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %327 = neura.grant_predicate %299, %325 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %328 = neura.grant_predicate %301, %325 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %329 = neura.grant_predicate %307, %325 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %330 = neura.grant_predicate %293, %325 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %331 = neura.grant_predicate %297, %325 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %332 = neura.grant_predicate %305, %325 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %333 = neura.grant_predicate %291, %325 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %334 = neura.grant_predicate %289, %325 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %335 = neura.grant_predicate %303, %325 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %336 = neura.grant_predicate %287, %325 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %337 = neura.grant_predicate %285, %325 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %338 = neura.grant_predicate %283, %325 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %339 = neura.load_indexed %326[%327, %328 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %340 = "neura.mul"(%339, %329) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %341 = "neura.div"(%340, %330) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %341 to %326[%327, %328 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %342 = "neura.add"(%328, %331) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %342 -> %166 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %332 -> %164 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %333 -> %162 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %334 -> %160 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %335 -> %158 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %327 -> %156 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %331 -> %154 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %326 -> %152 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %330 -> %150 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %336 -> %148 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %337 -> %146 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %338 -> %144 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %343 = neura.load_indexed %311[%312, %313, %314 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %344 = "neura.icmp"(%343, %315) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %345 = "neura.sel"(%344, %343, %315) : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %346 = "neura.add"(%312, %316) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %346 -> %308 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %345 -> %306 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %317 -> %304 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %311 -> %302 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %313 -> %300 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %314 -> %298 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %316 -> %296 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %318 -> %294 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %319 -> %292 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %320 -> %290 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %321 -> %288 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %322 -> %286 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %323 -> %284 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %324 -> %282 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %347 = neura.load_indexed %119[%120, %121 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %348 = "neura.add"(%122, %347) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %349 = "neura.add"(%121, %123) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %349 -> %116 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %348 -> %114 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %124 -> %112 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %119 -> %110 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %120 -> %108 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %123 -> %106 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %125 -> %104 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %126 -> %102 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %127 -> %100 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %128 -> %98 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %129 -> %96 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %130 -> %94 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %72 to %73[%74, %75 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %350 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %351 = neura.phi_start %82, %350 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %352 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %353 = neura.phi_start %81, %352 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %354 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %355 = neura.phi_start %80, %354 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %356 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %357 = neura.phi_start %76, %356 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %358 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %359 = neura.phi_start %72, %358 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %360 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %361 = neura.phi_start %79, %360 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %362 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %363 = neura.phi_start %73, %362 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %364 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %365 = neura.phi_start %75, %364 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %366 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %367 = neura.phi_start %74, %366 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %368 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %369 = neura.phi_start %78, %368 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %370 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %371 = neura.phi_start %77, %370 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %372 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %373 = neura.phi_start %76, %372 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %374 = "neura.icmp"(%373, %371) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %375 = neura.grant_predicate %369, %374 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %376 = neura.grant_predicate %367, %374 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %377 = neura.grant_predicate %365, %374 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %378 = neura.grant_predicate %373, %374 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %379 = neura.grant_predicate %363, %374 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %380 = neura.grant_predicate %361, %374 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %381 = neura.grant_predicate %371, %374 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %382 = neura.grant_predicate %359, %374 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %383 = neura.grant_predicate %357, %374 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %384 = neura.grant_predicate %355, %374 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %385 = neura.grant_predicate %353, %374 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %386 = neura.grant_predicate %351, %374 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %387 = "neura.not"(%374) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %388 = neura.grant_predicate %365, %387 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %389 = neura.grant_predicate %361, %387 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %390 = neura.grant_predicate %371, %387 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %391 = neura.grant_predicate %359, %387 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %392 = neura.grant_predicate %363, %387 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %393 = neura.grant_predicate %367, %387 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %394 = neura.grant_predicate %357, %387 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %395 = neura.grant_predicate %369, %387 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %396 = neura.grant_predicate %355, %387 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %397 = neura.grant_predicate %353, %387 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %398 = neura.grant_predicate %351, %387 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %399 = "neura.add"(%388, %389) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %399 -> %69 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %390 -> %67 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %391 -> %65 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %392 -> %63 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %393 -> %61 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %394 -> %59 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %395 -> %57 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %389 -> %55 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %396 -> %53 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %397 -> %51 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %398 -> %49 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %400 = neura.load_indexed %375[%376, %377, %378 : !neura.data<i64, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %401 = neura.load_indexed %379[%376, %377 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %402 = "neura.add"(%401, %400) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %402 to %379[%376, %377 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %403 = "neura.add"(%378, %380) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %403 -> %372 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %381 -> %370 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %375 -> %368 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %376 -> %366 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %377 -> %364 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %379 -> %362 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %380 -> %360 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %382 -> %358 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %383 -> %356 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %384 -> %354 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %385 -> %352 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %386 -> %350 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %404 = "neura.constant"() <{value = true}> : () -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %405 = "neura.grant_once"(%404) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     "neura.return"(%405) : (!neura.data<i1, i1>) -> ()
// CTRL2DATA-NEXT:   }
// CTRL2DATA-NEXT: }
