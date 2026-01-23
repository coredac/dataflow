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
// RUN: --promote-input-arg-to-const \
// RUN: --canonicalize-return \
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

// CHECK:   func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
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

// CTRL2DATA:        func.func @_Z14complex_nestedPA32_A32_iPS_(%arg0: memref<?x32x32xi32>, %arg1: memref<?x32xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
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
// CTRL2DATA-NEXT:     %39 = "neura.not"(%38) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %40 = neura.grant_predicate %33, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %41 = neura.grant_predicate %35, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %31, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %43 = neura.grant_predicate %29, %38 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %37, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %45 = neura.grant_predicate %27, %38 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %25, %38 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %47 = neura.grant_predicate %23, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %21, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %49 = neura.grant_predicate %19, %38 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %39, %39 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     neura.return_void %50 : !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %51 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %52 = neura.phi_start %49, %51 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %53 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %54 = neura.phi_start %48, %53 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %55 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %56 = neura.phi_start %47, %55 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %57 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %58 = neura.phi_start %46, %57 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %60 = neura.phi_start %45, %59 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %61 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %62 = neura.phi_start %40, %61 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %63 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %64 = neura.phi_start %44, %63 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %65 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %66 = neura.phi_start %43, %65 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %67 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %68 = neura.phi_start %42, %67 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %69 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %70 = neura.phi_start %41, %69 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %71 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %72 = neura.phi_start %40, %71 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %73 = "neura.icmp"(%72, %70) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %74 = neura.grant_predicate %68, %73 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %75 = neura.grant_predicate %66, %73 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %76 = neura.grant_predicate %64, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %77 = neura.grant_predicate %72, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %78 = neura.grant_predicate %62, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %79 = neura.grant_predicate %70, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %80 = neura.grant_predicate %60, %73 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %81 = neura.grant_predicate %58, %73 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %82 = neura.grant_predicate %56, %73 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %83 = neura.grant_predicate %54, %73 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %84 = neura.grant_predicate %52, %73 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %85 = "neura.not"(%73) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %86 = neura.grant_predicate %62, %85 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %87 = neura.grant_predicate %68, %85 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %88 = neura.grant_predicate %70, %85 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %89 = neura.grant_predicate %66, %85 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %90 = neura.grant_predicate %64, %85 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %91 = neura.grant_predicate %58, %85 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %92 = neura.grant_predicate %56, %85 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %93 = neura.grant_predicate %54, %85 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %94 = neura.grant_predicate %60, %85 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %95 = neura.grant_predicate %52, %85 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %96 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %97 = neura.phi_start %87, %96 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %98 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %99 = neura.phi_start %95, %98 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %100 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %101 = neura.phi_start %94, %100 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %102 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %103 = neura.phi_start %93, %102 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %104 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %105 = neura.phi_start %86, %104 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %106 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %107 = neura.phi_start %92, %106 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %108 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %109 = neura.phi_start %91, %108 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %110 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %111 = neura.phi_start %90, %110 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %112 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %113 = neura.phi_start %89, %112 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %114 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %115 = neura.phi_start %88, %114 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %116 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %117 = neura.phi_start %87, %116 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %118 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %119 = neura.phi_start %86, %118 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %120 = "neura.icmp"(%119, %115) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %121 = neura.grant_predicate %113, %120 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %122 = neura.grant_predicate %111, %120 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %123 = neura.grant_predicate %119, %120 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %124 = neura.grant_predicate %117, %120 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %125 = neura.grant_predicate %109, %120 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %126 = neura.grant_predicate %115, %120 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %127 = neura.grant_predicate %107, %120 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %128 = neura.grant_predicate %105, %120 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %129 = neura.grant_predicate %103, %120 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %130 = neura.grant_predicate %101, %120 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %131 = neura.grant_predicate %99, %120 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %132 = neura.grant_predicate %97, %120 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %133 = "neura.not"(%120) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %134 = neura.grant_predicate %117, %133 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %135 = neura.grant_predicate %107, %133 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %136 = neura.grant_predicate %105, %133 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %137 = neura.grant_predicate %115, %133 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %138 = neura.grant_predicate %103, %133 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %139 = neura.grant_predicate %101, %133 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %140 = neura.grant_predicate %111, %133 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %141 = neura.grant_predicate %109, %133 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %142 = neura.grant_predicate %113, %133 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %143 = neura.grant_predicate %99, %133 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %144 = neura.grant_predicate %97, %133 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %145 = "neura.div"(%134, %135) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %146 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %147 = neura.phi_start %135, %146 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %148 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %149 = neura.phi_start %144, %148 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %150 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %151 = neura.phi_start %145, %150 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %152 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %153 = neura.phi_start %143, %152 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %154 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %155 = neura.phi_start %142, %154 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %156 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %157 = neura.phi_start %141, %156 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %158 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %159 = neura.phi_start %140, %158 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %160 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %161 = neura.phi_start %139, %160 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %162 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %163 = neura.phi_start %138, %162 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %164 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %165 = neura.phi_start %136, %164 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %166 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %167 = neura.phi_start %137, %166 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %168 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %169 = neura.phi_start %136, %168 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %170 = "neura.icmp"(%169, %167) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %171 = neura.grant_predicate %165, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %172 = neura.grant_predicate %163, %170 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %173 = neura.grant_predicate %167, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %174 = neura.grant_predicate %161, %170 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %175 = neura.grant_predicate %169, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %176 = neura.grant_predicate %159, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %177 = neura.grant_predicate %157, %170 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %178 = neura.grant_predicate %155, %170 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %179 = neura.grant_predicate %153, %170 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %180 = neura.grant_predicate %151, %170 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %181 = neura.grant_predicate %149, %170 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %182 = neura.grant_predicate %147, %170 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %183 = "neura.not"(%170) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %184 = neura.grant_predicate %165, %183 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %185 = neura.grant_predicate %167, %183 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %186 = neura.grant_predicate %155, %183 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %187 = neura.grant_predicate %159, %183 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %188 = neura.grant_predicate %151, %183 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %189 = neura.grant_predicate %157, %183 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %190 = neura.grant_predicate %149, %183 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %191 = neura.grant_predicate %161, %183 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %192 = neura.grant_predicate %147, %183 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %193 = neura.grant_predicate %163, %183 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %194 = neura.grant_predicate %153, %183 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %195 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %196 = neura.phi_start %194, %195 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %197 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %198 = neura.phi_start %193, %197 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %199 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %200 = neura.phi_start %192, %199 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %201 = neura.reserve : !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %202 = neura.phi_start %191, %201 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<memref<?x32x32xi32>, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %203 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %204 = neura.phi_start %190, %203 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %205 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %206 = neura.phi_start %184, %205 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %207 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %208 = neura.phi_start %189, %207 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %209 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %210 = neura.phi_start %188, %209 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %211 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %212 = neura.phi_start %187, %211 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %213 = neura.reserve : !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %214 = neura.phi_start %186, %213 : !neura.data<memref<?x32xi32>, i1>, !neura.data<memref<?x32xi32>, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %215 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %216 = neura.phi_start %185, %215 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %217 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %218 = neura.phi_start %184, %217 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %219 = "neura.icmp"(%218, %216) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %220 = neura.grant_predicate %214, %219 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %221 = neura.grant_predicate %212, %219 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %222 = neura.grant_predicate %218, %219 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %223 = neura.grant_predicate %210, %219 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %224 = neura.grant_predicate %208, %219 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %225 = neura.grant_predicate %216, %219 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %226 = neura.grant_predicate %206, %219 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %227 = neura.grant_predicate %204, %219 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %228 = neura.grant_predicate %202, %219 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %229 = neura.grant_predicate %200, %219 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %230 = neura.grant_predicate %198, %219 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %231 = neura.grant_predicate %196, %219 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %232 = "neura.not"(%219) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %233 = neura.grant_predicate %212, %232 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %234 = neura.grant_predicate %208, %232 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %235 = neura.grant_predicate %216, %232 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %236 = neura.grant_predicate %206, %232 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %237 = neura.grant_predicate %204, %232 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %238 = neura.grant_predicate %214, %232 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %239 = neura.grant_predicate %202, %232 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %240 = neura.grant_predicate %200, %232 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %241 = neura.grant_predicate %198, %232 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %242 = neura.grant_predicate %196, %232 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %243 = "neura.add"(%233, %234) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %243 -> %36 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %235 -> %34 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %236 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %237 -> %30 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %238 -> %28 : !neura.data<memref<?x32xi32>, i1> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %239 -> %26 : !neura.data<memref<?x32x32xi32>, i1> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %234 -> %24 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %240 -> %22 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %241 -> %20 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %242 -> %18 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %244 = neura.load_indexed %220[%221, %222 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x32xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %245 = "neura.icmp"(%244, %223) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %246 = neura.grant_predicate %223, %245 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %247 = neura.grant_predicate %220, %245 : !neura.data<memref<?x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32xi32>, i1>
// CTRL2DATA-NEXT:     %248 = neura.grant_predicate %221, %245 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %249 = neura.grant_predicate %222, %245 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %250 = neura.grant_predicate %224, %245 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %251 = neura.grant_predicate %225, %245 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %252 = neura.grant_predicate %226, %245 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %253 = neura.grant_predicate %227, %245 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %254 = neura.grant_predicate %228, %245 : !neura.data<memref<?x32x32xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x32x32xi32>, i1>
// CTRL2DATA-NEXT:     %255 = neura.grant_predicate %229, %245 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %256 = neura.grant_predicate %230, %245 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %257 = neura.grant_predicate %231, %245 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %258 = "neura.not"(%245) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %259 = neura.grant_predicate %222, %258 : !neura.data<i64, i1
