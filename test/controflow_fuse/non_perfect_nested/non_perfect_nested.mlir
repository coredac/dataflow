// RUN: mlir-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm -o %t-llvm.mlir
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura | FileCheck %s
// RUN: mlir-neura-opt %t-llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --neura-canonicalize --leverage-predicated-value --transform-ctrl-to-data-flow | FileCheck %s -check-prefix=CTRL2DATA

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
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 4 : index}> : () -> index
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = 3 : index}> : () -> index
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = 2 : index}> : () -> index
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> index
// CHECK-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> index
// CHECK-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 100 : i32}> : () -> i32
// CHECK-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 3 : i32}> : () -> i32
// CHECK-NEXT:     %7 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> i32
// CHECK-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1000 : i32}> : () -> i32
// CHECK-NEXT:     %9 = "neura.constant"() <{predicate = true, value = -1000 : i32}> : () -> i32
// CHECK-NEXT:     %10 = "neura.constant"() <{predicate = true, value = 128 : i32}> : () -> i32
// CHECK-NEXT:     %11 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> index
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

// CTRL2DATA:   func.func @_Z29non_perfect_extra_computationPA128_iS0_(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi32>) attributes {accelerator = "neura", llvm.linkage = #llvm.linkage<external>} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %2 = "neura.grant_once"(%0) : (!neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %3 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %4 = "neura.grant_always"(%3) : (!neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%3) : (!neura.data<memref<?x128xi32>, i1>) -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 4 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %8 = "neura.grant_once"(%6) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 3 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %10 = "neura.grant_always"(%9) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%9) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 2 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %13 = "neura.grant_always"(%12) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %14 = "neura.grant_once"(%12) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %15 = "neura.constant"() <{predicate = true, value = 1 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %16 = "neura.grant_always"(%15) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %17 = "neura.grant_once"(%15) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %18 = "neura.constant"() <{predicate = true, value = 128 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %19 = "neura.grant_always"(%18) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %20 = "neura.grant_once"(%18) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %21 = "neura.constant"() <{predicate = true, value = 100 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %22 = "neura.grant_always"(%21) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %23 = "neura.grant_once"(%21) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %24 = "neura.constant"() <{predicate = true, value = 3 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %25 = "neura.grant_always"(%24) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %26 = "neura.grant_once"(%24) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %27 = "neura.constant"() <{predicate = true, value = 2 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %28 = "neura.grant_always"(%27) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %29 = "neura.grant_once"(%27) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %30 = "neura.constant"() <{predicate = true, value = 1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %31 = "neura.grant_always"(%30) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %32 = "neura.grant_once"(%30) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %33 = "neura.constant"() <{predicate = true, value = -1000 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %34 = "neura.grant_always"(%33) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %35 = "neura.grant_once"(%33) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %36 = "neura.constant"() <{predicate = true, value = 128 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %37 = "neura.grant_always"(%36) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %38 = "neura.grant_once"(%36) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = "neura.constant"() <{predicate = true, value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %40 = "neura.grant_always"(%39) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %41 = "neura.grant_once"(%39) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %42 = "neura.constant"() <{predicate = true, value = 0 : index}> : () -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %43 = "neura.grant_always"(%42) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %44 = "neura.grant_once"(%42) : (!neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %45 = "neura.cast"(%42) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %46 = "neura.grant_once"(%45) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %47 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %48 = "neura.phi"(%47, %20) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %49 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %50 = "neura.phi"(%49, %46) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %51 = "neura.cast"(%50) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %52 = "neura.icmp"(%51, %48) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %53 = "neura.not"(%52) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %54 = neura.grant_predicate %51, %52 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %29, %52 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %56 = neura.grant_predicate %41, %52 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %57 = neura.grant_predicate %26, %52 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %58 = neura.grant_predicate %44, %52 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %59 = neura.grant_predicate %32, %52 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %60 = neura.grant_predicate %35, %52 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %61 = "neura.cast"(%54) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %62 = "neura.div"(%61, %55) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %63 = "neura.mul"(%55, %62) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %64 = "neura.sub"(%61, %63) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %65 = "neura.icmp"(%64, %56) <{cmpType = "eq"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %66 = "neura.sel"(%55, %57, %65) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %67 = "neura.cast"(%58) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %68 = neura.reserve : !neura.data<index, i1>
// CTRL2DATA-NEXT:     %69 = "neura.phi"(%68, %20) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %70 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %71 = "neura.phi"(%70, %56) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %72 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %73 = "neura.phi"(%72, %60) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %74 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %75 = "neura.phi"(%74, %59) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %76 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %77 = "neura.phi"(%76, %67) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %78 = "neura.cast"(%77) <{cast_type = "int_to_index"}> : (!neura.data<i64, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %79 = "neura.icmp"(%78, %69) <{cmpType = "slt"}> : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %80 = neura.grant_predicate %2, %79 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %81 = neura.grant_predicate %51, %79 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %82 = neura.grant_predicate %78, %79 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %83 = neura.grant_predicate %66, %79 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %84 = neura.grant_predicate %5, %79 : !neura.data<memref<?x128xi32>, i1>, !neura.data<i1, i1> -> !neura.data<memref<?x128xi32>, i1>
// CTRL2DATA-NEXT:     %85 = neura.grant_predicate %71, %79 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %86 = neura.grant_predicate %73, %79 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %87 = neura.grant_predicate %75, %79 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %88 = neura.grant_predicate %17, %79 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %89 = neura.grant_predicate %20, %79 : !neura.data<index, i1>, !neura.data<i1, i1> -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %90 = "neura.not"(%79) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %91 = neura.grant_predicate %71, %90 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %92 = neura.grant_predicate %38, %90 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %93 = neura.grant_predicate %73, %90 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %94 = neura.grant_predicate %75, %90 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %95 = neura.grant_predicate %41, %90 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %96 = neura.load_indexed %80[%81, %82 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %97 = "neura.mul"(%96, %83) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %97 to %84[%81, %82 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %98 = neura.load_indexed %80[%81, %82 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %99 = "neura.add"(%85, %98) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %100 = "neura.icmp"(%98, %86) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %101 = "neura.sel"(%98, %86, %100) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %102 = "neura.icmp"(%98, %87) <{cmpType = "slt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %103 = "neura.sel"(%98, %87, %102) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i1, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %104 = "neura.add"(%82, %88) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %105 = "neura.cast"(%104) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %105 -> %76 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %103 -> %74 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %101 -> %72 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %99 -> %70 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %89 -> %68 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %106 = "neura.div"(%91, %92) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %107 = "neura.sub"(%93, %94) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %108 = "neura.icmp"(%107, %95) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %109 = neura.grant_predicate %106, %108 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %110 = neura.grant_predicate %23, %108 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %111 = neura.grant_predicate %107, %108 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %112 = "neura.not"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %113 = neura.grant_predicate %106, %112 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %114 = "neura.mul"(%109, %110) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %115 = "neura.div"(%114, %111) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %116 = "neura.phi"(%115, %113) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %106 to %5[%51, %44 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %73 to %5[%51, %17 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %75 to %5[%51, %14 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %116 to %5[%51, %11 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.store_indexed %107 to %5[%51, %8 : !neura.data<index, i1>, !neura.data<index, i1>] !neura.data<memref<?x128xi32>, i1> : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %117 = "neura.add"(%51, %17) : (!neura.data<index, i1>, !neura.data<index, i1>) -> !neura.data<index, i1>
// CTRL2DATA-NEXT:     %118 = "neura.cast"(%117) <{cast_type = "index_to_int"}> : (!neura.data<index, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %118 -> %49 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %20 -> %47 : !neura.data<index, i1> !neura.data<index, i1>
// CTRL2DATA-NEXT:     "neura.return"() : () -> ()
// CTRL2DATA-NEXT:   }