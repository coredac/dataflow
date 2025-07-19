// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

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

// CHECK: func.func @_Z29non_perfect_extra_computationPA128_iS0_(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
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
// CHECK-NEXT:     %22 = "neura.sel"(%7, %6, %21) : (i32, i32, i1) -> i32
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
// CHECK-NEXT:     %35 = "neura.sel"(%32, %26, %34) : (i32, i32, i1) -> i32
// CHECK-NEXT:     %36 = "neura.icmp"(%32, %25) <{cmpType = "slt"}> : (i32, i32) -> i1
// CHECK-NEXT:     %37 = "neura.sel"(%32, %25, %36) : (i32, i32, i1) -> i32
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

// CTRL2DATA: func.func @_Z29non_perfect_extra_computationPA128_iS0_(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = 4 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = 3 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_always"(%2) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 2 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_always"(%4) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 128 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_always"(%8) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = 100 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_always"(%10) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{value = 3 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_always"(%12) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %14 = "neura.constant"() <{value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %15 = "neura.grant_always"(%14) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %16 = "neura.constant"() <{value = 1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_always"(%16) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %18 = "neura.grant_once"(%16) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = "neura.constant"() <{value = -1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = "neura.grant_always"(%19) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %21 = "neura.grant_once"(%19) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = "neura.constant"() <{value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.grant_always"(%22) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %25 = "neura.grant_always"(%24) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %26 = "neura.grant_once"(%24) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %27 = "neura.constant"() <{value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %28 = "neura.grant_always"(%27) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %29 = "neura.cast"(%27) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %30 = "neura.grant_once"(%29) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = "neura.phi"(%31, %30) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %33 = "neura.cast"(%32) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %34 = "neura.icmp"(%33, %9) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %35 = "neura.not"(%34) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %36 = neura.grant_predicate %33, %34 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %37 = "neura.cast"(%36) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %38 = neura.grant_predicate %15, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = "neura.div"(%37, %38) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %40 = "neura.mul"(%38, %39) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %41 = "neura.sub"(%37, %40) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %42 = neura.grant_predicate %25, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %43 = "neura.icmp"(%41, %42) <{cmpType = "eq"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %44 = neura.grant_predicate %13, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %45 = "neura.sel"(%38, %44, %43) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %46 = neura.grant_predicate %28, %34 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %47 = "neura.cast"(%46) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %48 = neura.grant_predicate %18, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %49 = neura.grant_predicate %21, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %50 = neura.grant_predicate %26, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %51 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %52 = "neura.phi"(%51, %50) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %53 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %54 = "neura.phi"(%53, %49) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %55 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %56 = "neura.phi"(%55, %48) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %57 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %58 = "neura.phi"(%57, %47) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %59 = "neura.cast"(%58) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %60 = "neura.icmp"(%59, %9) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %61 = "neura.not"(%60) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %62 = neura.grant_predicate %33, %60 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %63 = neura.grant_predicate %59, %60 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %64 = neura.load_indexed %arg0[%62, %63 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %65 = neura.grant_predicate %45, %60 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %66 = "neura.mul"(%64, %65) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %66 to %arg1[%62, %63 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %67 = neura.load_indexed %arg0[%62, %63 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %68 = "neura.add"(%52, %67) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %69 = "neura.icmp"(%67, %54) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %70 = "neura.sel"(%67, %54, %69) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %71 = "neura.icmp"(%67, %56) <{cmpType = "slt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %72 = "neura.sel"(%67, %56, %71) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %73 = neura.grant_predicate %7, %60 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %74 = "neura.add"(%63, %73) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %75 = "neura.cast"(%74) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %75 -> %57 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %72 -> %55 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %70 -> %53 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %68 -> %51 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %76 = neura.grant_predicate %23, %61 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %77 = "neura.div"(%52, %76) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %78 = "neura.sub"(%54, %56) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %79 = neura.grant_predicate %25, %61 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %80 = "neura.icmp"(%78, %79) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %81 = "neura.not"(%80) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %82 = neura.grant_predicate %77, %80 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %83 = neura.grant_predicate %11, %80 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %84 = "neura.mul"(%82, %83) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %85 = neura.grant_predicate %78, %80 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %86 = "neura.div"(%84, %85) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %87 = neura.grant_predicate %77, %81 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %88 = "neura.phi"(%86, %87) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %77 to %arg1[%33, %28 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %54 to %arg1[%33, %7 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %56 to %arg1[%33, %5 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %88 to %arg1[%33, %3 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %78 to %arg1[%33, %1 : !neura.data<index, i1>, !neura.data<index, i1>] memref<?x128xi32> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %89 = "neura.add"(%33, %7) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %90 = "neura.cast"(%89) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %90 -> %31 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }
