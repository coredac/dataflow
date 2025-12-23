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
  func.func @_Z29non_perfect_extra_computationPA128_iS0_(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c100_i32 = arith.constant 100 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %c-1000_i32 = arith.constant -1000 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.remsi %0, %c2_i32 : i32
      %2 = arith.cmpi eq, %1, %c0_i32 : i32
      %3 = arith.select %2, %c2_i32, %c3_i32 : i32
      %4:3 = affine.for %arg3 = 0 to 128 iter_args(%arg4 = %c1000_i32, %arg5 = %c-1000_i32, %arg6 = %c0_i32) -> (i32, i32, i32) {
        %9 = affine.load %arg0[%arg2, %arg3] : memref<?x128xi32>
        %10 = arith.muli %9, %3 : i32
        affine.store %10, %arg1[%arg2, %arg3] : memref<?x128xi32>
        %11 = affine.load %arg0[%arg2, %arg3] : memref<?x128xi32>
        %12 = arith.addi %arg6, %11 : i32
        %13 = arith.cmpi sgt, %11, %arg5 : i32
        %14 = arith.select %13, %11, %arg5 : i32
        %15 = arith.cmpi slt, %11, %arg4 : i32
        %16 = arith.select %15, %11, %arg4 : i32
        affine.yield %16, %14, %12 : i32, i32, i32
      }
      %5 = arith.divsi %4#2, %c128_i32 : i32
      %6 = arith.subi %4#1, %4#0 : i32
      %7 = arith.cmpi sgt, %6, %c0_i32 : i32
      %8 = scf.if %7 -> (i32) {
        %9 = arith.muli %5, %c100_i32 : i32
        %10 = arith.divsi %9, %6 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %5 : i32
      }
      affine.store %5, %arg1[%arg2, 0] : memref<?x128xi32>
      affine.store %4#1, %arg1[%arg2, 1] : memref<?x128xi32>
      affine.store %4#0, %arg1[%arg2, 2] : memref<?x128xi32>
      affine.store %8, %arg1[%arg2, 3] : memref<?x128xi32>
      affine.store %6, %arg1[%arg2, 4] : memref<?x128xi32>
    }
    return
  }
}

// CHECK:   func.func @_Z29non_perfect_extra_computationPA128_iS0_(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = 4 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{value = 3 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 128 : index}> : () -> index
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 100 : i32}> : () -> i32
// CHECK-NEXT:     %6 = "neura.constant"() <{value = 3 : i32}> : () -> i32
// CHECK-NEXT:     %7 = "neura.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:     %8 = "neura.constant"() <{value = 1000 : i32}> : () -> i32
// CHECK-NEXT:     %9 = "neura.constant"() <{value = -1000 : i32}> : () -> i32
// CHECK-NEXT:     %10 = "neura.constant"() <{value = 128 : i32}> : () -> i32
// CHECK-NEXT:     %11 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %12 = "neura.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:     %13 = "neura.cast"(%12) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %13 : i64 to ^bb1
// CHECK-NEXT:   ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb9
// CHECK-NEXT:     %15 = "neura.cast"(%14) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %16 = "neura.icmp"(%15, %4) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %16 : i1 then to ^bb2 else to ^bb10
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %17 = "neura.cast"(%15) <{cast_type = "index_to_int"}> : (index) -> i32
// CHECK-NEXT:     %18 = "neura.div"(%17, %7) : (i32, i32) -> i32
// CHECK-NEXT:     %19 = "neura.mul"(%7, %18) : (i32, i32) -> i32
// CHECK-NEXT:     %20 = "neura.sub"(%17, %19) : (i32, i32) -> i32
// CHECK-NEXT:     %21 = "neura.icmp"(%20, %11) <{cmpType = "eq"}> : (i32, i32) -> i1
// CHECK-NEXT:     %22 = "neura.sel"(%21, %7, %6) : (i1, i32, i32) -> i32
// CHECK-NEXT:     %23 = "neura.cast"(%12) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %23, %8, %9, %11 : i64, i32, i32, i32 to ^bb3
// CHECK-NEXT:   ^bb3(%24: i64, %25: i32, %26: i32, %27: i32):  // 2 preds: ^bb2, ^bb4
// CHECK-NEXT:     %28 = "neura.cast"(%24) <{cast_type = "int_to_index"}> : (i64) -> index
// CHECK-NEXT:     %29 = "neura.icmp"(%28, %4) <{cmpType = "slt"}> : (index, index) -> i1
// CHECK-NEXT:     neura.cond_br %29 : i1 then to ^bb4 else to ^bb5
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     %30 = neura.load_indexed %arg0[%15, %28 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     %31 = "neura.mul"(%30, %22) : (i32, i32) -> i32
// CHECK-NEXT:     neura.store_indexed %31 to %arg1[%15, %28 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     %32 = neura.load_indexed %arg0[%15, %28 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     %33 = "neura.add"(%27, %32) : (i32, i32) -> i32
// CHECK-NEXT:     %34 = "neura.icmp"(%32, %26) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     %35 = "neura.sel"(%34, %32, %26) : (i1, i32, i32) -> i32
// CHECK-NEXT:     %36 = "neura.icmp"(%32, %25) <{cmpType = "slt"}> : (i32, i32) -> i1
// CHECK-NEXT:     %37 = "neura.sel"(%36, %32, %25) : (i1, i32, i32) -> i32
// CHECK-NEXT:     %38 = "neura.add"(%28, %3) : (index, index) -> index
// CHECK-NEXT:     %39 = "neura.cast"(%38) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %39, %37, %35, %33 : i64, i32, i32, i32 to ^bb3
// CHECK-NEXT:   ^bb5:  // pred: ^bb3
// CHECK-NEXT:     %40 = "neura.div"(%27, %10) : (i32, i32) -> i32
// CHECK-NEXT:     %41 = "neura.sub"(%26, %25) : (i32, i32) -> i32
// CHECK-NEXT:     %42 = "neura.icmp"(%41, %11) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %42 : i1 then to ^bb6 else to ^bb7
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     %43 = "neura.mul"(%40, %5) : (i32, i32) -> i32
// CHECK-NEXT:     %44 = "neura.div"(%43, %41) : (i32, i32) -> i32
// CHECK-NEXT:     neura.br %44 : i32 to ^bb8
// CHECK-NEXT:   ^bb7:  // pred: ^bb5
// CHECK-NEXT:     neura.br %40 : i32 to ^bb8
// CHECK-NEXT:   ^bb8(%45: i32):  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:     neura.br to ^bb9
// CHECK-NEXT:   ^bb9:  // pred: ^bb8
// CHECK-NEXT:     neura.store_indexed %40 to %arg1[%15, %12 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     neura.store_indexed %26 to %arg1[%15, %3 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     neura.store_indexed %25 to %arg1[%15, %2 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     neura.store_indexed %45 to %arg1[%15, %1 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     neura.store_indexed %41 to %arg1[%15, %0 : index, index] memref<?x128xi32> : i32
// CHECK-NEXT:     %46 = "neura.add"(%15, %3) : (index, index) -> index
// CHECK-NEXT:     %47 = "neura.cast"(%46) <{cast_type = "index_to_int"}> : (index) -> i64
// CHECK-NEXT:     neura.br %47 : i64 to ^bb1
// CHECK-NEXT:   ^bb10:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }


// CTRL2DATA:        func.func @_Z29non_perfect_extra_computationPA128_iS0_(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi32>) attributes {accelerator = "neura", dataflow_mode = "predicate", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 4 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 3 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 2 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{value = 128 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = "neura.constant"() <{value = 100 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %15 = "neura.grant_once"(%14) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %16 = "neura.constant"() <{value = 3 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %18 = "neura.constant"() <{value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = "neura.grant_once"(%18) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = "neura.constant"() <{value = 1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.grant_once"(%20) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = "neura.constant"() <{value = -1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.grant_once"(%22) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = "neura.constant"() <{value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %25 = "neura.grant_once"(%24) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %26 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %27 = "neura.grant_once"(%26) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %28 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = "neura.grant_once"(%28) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.phi_start %30, %5 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = neura.phi_start %32, %7 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %34 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %35 = neura.phi_start %34, %9 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %37 = neura.phi_start %36, %15 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %38 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = neura.phi_start %38, %25 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %40 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %41 = neura.phi_start %40, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %42 = neura.reserve : !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %43 = neura.phi_start %42, %3 : !neura.data<memref<?x128xi32>, i1>, !neura.data<memref<?x128xi32>, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %44 = neura.reserve : !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %45 = neura.phi_start %44, %1 : !neura.data<memref<?x128xi32>, i1>, !neura.data<memref<?x128xi32>, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %46 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %47 = neura.phi_start %46, %23 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %48 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %49 = neura.phi_start %48, %21 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %50 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %51 = neura.phi_start %50, %29 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %52 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %53 = neura.phi_start %52, %17 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %54 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %55 = neura.phi_start %54, %27 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %56 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %57 = neura.phi_start %56, %19 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %58 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = neura.phi_start %58, %13 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %60 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %61 = neura.phi_start %60, %29 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %62 = "neura.icmp"(%61, %59) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %63 = neura.grant_predicate %61, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %64 = neura.grant_predicate %57, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %65 = neura.grant_predicate %55, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %66 = neura.grant_predicate %53, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %67 = neura.grant_predicate %51, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %68 = neura.grant_predicate %49, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %69 = neura.grant_predicate %47, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %70 = neura.grant_predicate %59, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %71 = neura.grant_predicate %45, %62 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %72 = neura.grant_predicate %43, %62 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %73 = neura.grant_predicate %41, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %74 = neura.grant_predicate %39, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %75 = neura.grant_predicate %37, %62 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %76 = neura.grant_predicate %35, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %77 = neura.grant_predicate %33, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %78 = neura.grant_predicate %31, %62 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %79 = "neura.cast"(%63) <{cast_type = "i64_to_i32"}> : (!neura.data<i64, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %80 = "neura.div"(%79, %64) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %81 = "neura.mul"(%64, %80) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %82 = "neura.sub"(%79, %81) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %83 = "neura.icmp"(%82, %65) <{cmpType = "eq"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %84 = "neura.sel"(%83, %64, %66) : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %85 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %86 = neura.phi_start %85, %69 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %87 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %88 = neura.phi_start %87, %68 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %89 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %90 = neura.phi_start %89, %66 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %91 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %92 = neura.phi_start %91, %64 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %93 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %94 = neura.phi_start %93, %78 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %95 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %96 = neura.phi_start %95, %77 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %97 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %98 = neura.phi_start %97, %76 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %99 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %100 = neura.phi_start %99, %67 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %101 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %102 = neura.phi_start %101, %75 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %103 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %104 = neura.phi_start %103, %65 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %105 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %106 = neura.phi_start %105, %74 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %107 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %108 = neura.phi_start %107, %73 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %109 = neura.reserve : !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %110 = neura.phi_start %109, %72 : !neura.data<memref<?x128xi32>, i1>, !neura.data<memref<?x128xi32>, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %111 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %112 = neura.phi_start %111, %84 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %113 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %114 = neura.phi_start %113, %63 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %115 = neura.reserve : !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %116 = neura.phi_start %115, %71 : !neura.data<memref<?x128xi32>, i1>, !neura.data<memref<?x128xi32>, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %117 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %118 = neura.phi_start %117, %70 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %119 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %120 = neura.phi_start %119, %65 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %121 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %122 = neura.phi_start %121, %69 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %123 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %124 = neura.phi_start %123, %68 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %125 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %126 = neura.phi_start %125, %67 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %127 = "neura.icmp"(%126, %118) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %128 = neura.grant_predicate %116, %127 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %129 = neura.grant_predicate %114, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %130 = neura.grant_predicate %126, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %131 = neura.grant_predicate %112, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %132 = neura.grant_predicate %110, %127 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %133 = neura.grant_predicate %120, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %134 = neura.grant_predicate %122, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %135 = neura.grant_predicate %124, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %136 = neura.grant_predicate %108, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %137 = neura.grant_predicate %118, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %138 = neura.grant_predicate %106, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %139 = neura.grant_predicate %104, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %140 = neura.grant_predicate %102, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %141 = neura.grant_predicate %100, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %142 = neura.grant_predicate %98, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %143 = neura.grant_predicate %96, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %144 = neura.grant_predicate %94, %127 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %145 = neura.grant_predicate %92, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %146 = neura.grant_predicate %90, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %147 = neura.grant_predicate %88, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %148 = neura.grant_predicate %86, %127 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %149 = "neura.not"(%127) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %150 = neura.grant_predicate %120, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %151 = neura.grant_predicate %106, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %152 = neura.grant_predicate %122, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %153 = neura.grant_predicate %124, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %154 = neura.grant_predicate %104, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %155 = neura.grant_predicate %102, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %156 = neura.grant_predicate %110, %149 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %157 = neura.grant_predicate %114, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %158 = neura.grant_predicate %100, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %159 = neura.grant_predicate %108, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %160 = neura.grant_predicate %98, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %161 = neura.grant_predicate %96, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %162 = neura.grant_predicate %94, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %163 = neura.grant_predicate %118, %149 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %164 = neura.grant_predicate %92, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %165 = neura.grant_predicate %90, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %166 = neura.grant_predicate %88, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %167 = neura.grant_predicate %86, %149 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %168 = neura.grant_predicate %116, %149 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %169 = "neura.div"(%150, %151) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %170 = "neura.sub"(%152, %153) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %171 = "neura.icmp"(%170, %154) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %172 = neura.grant_predicate %169, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %173 = neura.grant_predicate %155, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %174 = neura.grant_predicate %170, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %175 = neura.grant_predicate %156, %171 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %176 = neura.grant_predicate %157, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %177 = neura.grant_predicate %158, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %178 = neura.grant_predicate %152, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %179 = neura.grant_predicate %159, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %180 = neura.grant_predicate %153, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %181 = neura.grant_predicate %160, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %182 = neura.grant_predicate %161, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %183 = neura.grant_predicate %162, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %184 = neura.grant_predicate %163, %171 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %185 = neura.grant_predicate %164, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %186 = neura.grant_predicate %154, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %187 = neura.grant_predicate %165, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %188 = neura.grant_predicate %166, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %189 = neura.grant_predicate %167, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %190 = neura.grant_predicate %168, %171 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %191 = neura.grant_predicate %151, %171 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %192 = "neura.not"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %193 = neura.grant_predicate %169, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %194 = neura.grant_predicate %156, %192 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %195 = neura.grant_predicate %157, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %196 = neura.grant_predicate %158, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %197 = neura.grant_predicate %152, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %198 = neura.grant_predicate %159, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %199 = neura.grant_predicate %153, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %200 = neura.grant_predicate %160, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %201 = neura.grant_predicate %161, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %202 = neura.grant_predicate %170, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %203 = neura.grant_predicate %162, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %204 = neura.grant_predicate %163, %192 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %205 = neura.grant_predicate %164, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %206 = neura.grant_predicate %154, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %207 = neura.grant_predicate %165, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %208 = neura.grant_predicate %166, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %209 = neura.grant_predicate %167, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %210 = neura.grant_predicate %168, %192 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %211 = neura.grant_predicate %151, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %212 = neura.grant_predicate %155, %192 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %213 = "neura.mul"(%172, %173) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %214 = "neura.div"(%213, %174) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %215 = "neura.phi"(%173, %212) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %216 = "neura.phi"(%191, %211) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %217 = "neura.phi"(%190, %210) : (!neura.data<memref<?x128xi32>, i1>, !neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %218 = "neura.phi"(%189, %209) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %219 = "neura.phi"(%188, %208) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %220 = "neura.phi"(%187, %207) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %221 = "neura.phi"(%186, %206) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %222 = "neura.phi"(%185, %205) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %223 = "neura.phi"(%184, %204) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %224 = "neura.phi"(%183, %203) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %225 = "neura.phi"(%174, %202) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %226 = "neura.phi"(%182, %201) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %227 = "neura.phi"(%181, %200) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %228 = "neura.phi"(%180, %199) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %229 = "neura.phi"(%179, %198) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %230 = "neura.phi"(%178, %197) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %231 = "neura.phi"(%177, %196) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %232 = "neura.phi"(%176, %195) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %233 = "neura.phi"(%175, %194) : (!neura.data<memref<?x128xi32>, i1>, !neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %234 = "neura.phi"(%172, %193) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %235 = "neura.phi"(%214, %193) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %234 to %233[%232, %231 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %230 to %233[%232, %229 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %228 to %233[%232, %227 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %235 to %233[%232, %226 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %225 to %233[%232, %224 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %236 = "neura.add"(%232, %229) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %236 -> %60 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %223 -> %58 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %222 -> %56 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %221 -> %54 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %220 -> %52 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %231 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %219 -> %48 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %218 -> %46 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %217 -> %44 : !neura.data<memref<?x128xi32>, i1> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %233 -> %42 : !neura.data<memref<?x128xi32>, i1> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %229 -> %40 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %216 -> %38 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %215 -> %36 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %227 -> %34 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %226 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %224 -> %30 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %237 = neura.load_indexed %128[%129, %130 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %238 = "neura.mul"(%237, %131) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %238 to %132[%129, %130 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %239 = neura.load_indexed %128[%129, %130 : !neura.data<i64, i1>, !neura.data<i64, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %240 = "neura.add"(%133, %239) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %241 = "neura.icmp"(%239, %134) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %242 = "neura.sel"(%241, %239, %134) : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %243 = "neura.icmp"(%239, %135) <{cmpType = "slt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %244 = "neura.sel"(%243, %239, %135) : (!neura.data<i1, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %245 = "neura.add"(%130, %136) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %245 -> %125 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %244 -> %123 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %242 -> %121 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %240 -> %119 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %137 -> %117 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %128 -> %115 : !neura.data<memref<?x128xi32>, i1> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %129 -> %113 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %131 -> %111 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %132 -> %109 : !neura.data<memref<?x128xi32>, i1> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %136 -> %107 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %138 -> %105 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %139 -> %103 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %140 -> %101 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %141 -> %99 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %142 -> %97 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %143 -> %95 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %144 -> %93 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %145 -> %91 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %146 -> %89 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %147 -> %87 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %148 -> %85 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }