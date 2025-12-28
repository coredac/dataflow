// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O1 -o %t-relu.ll relu.cpp
// RUN: mlir-translate --import-llvm %t-relu.ll -o %t-relu.mlir

// RUN: mlir-neura-opt %t-relu.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:  | FileCheck %s

// RUN: mlir-neura-opt %t-relu.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:  | FileCheck %s --check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %t-relu.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=customized" \
// RUN:  | FileCheck %s --check-prefix=MAPPING

// CHECK:      module attributes {{.*}}
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @input(dense<[1, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31]> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @output(dense<0> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("output[%d] = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %1 = llvm.mlir.addressof @input : !llvm.ptr
// CHECK-NEXT:     %2 = llvm.mlir.addressof @output : !llvm.ptr
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:     %6 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %7 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:     %8 = "neura.constant"() <{value = 32 : i64}> : () -> i64
// CHECK-NEXT:     "neura.memset"(%2, %3, %4) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:     neura.br %5 : i64 to ^bb1
// CHECK-NEXT:   ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:     %10 = "neura.gep"(%1, %9) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %11 = "neura.load"(%10) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %12 = "neura.icmp"(%11, %6) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %12 : i1 then to ^bb2 else to ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %13 = "neura.gep"(%2, %9) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %14 = "neura.load"(%13) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %15 = "neura.add"(%14, %11) : (i32, i32) -> i32
// CHECK-NEXT:     "neura.store"(%15, %13) : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:     neura.br to ^bb3
// CHECK-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:     %16 = "neura.add"(%9, %7) : (i64, i64) -> i64
// CHECK-NEXT:     %17 = "neura.icmp"(%16, %8) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT:     neura.cond_br %17 : i1 then %5 : i64 to ^bb5 else %16 : i64 to ^bb1
// CHECK-NEXT:   ^bb4:  // pred: ^bb5
// CHECK-NEXT:     "neura.return"(%6) : (i32) -> ()
// CHECK-NEXT:   ^bb5(%18: i64):  // 2 preds: ^bb3, ^bb5
// CHECK-NEXT:     %19 = "neura.constant"() <{value = 0 : i32}> : () -> index
// CHECK-NEXT:     %20 = "neura.gep"(%2, %19, %18) <{operandSegmentSizes = array<i32: 1, 2>}> : (!llvm.ptr, index, i64) -> !llvm.ptr
// CHECK-NEXT:     %21 = "neura.load"(%20) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %22 = "neura.cast"(%18) <{cast_type = "trunc"}> : (i64) -> i32
// CHECK-NEXT:     %23 = llvm.call tail @printf(%0, %22, %21) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32) -> i32
// CHECK-NEXT:     %24 = "neura.add"(%18, %7) : (i64, i64) -> i64
// CHECK-NEXT:     %25 = "neura.icmp"(%24, %8) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT:     neura.cond_br %25 : i1 then to ^bb4 else %24 : i64 to ^bb5
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !llvm.ptr
// CHECK-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> !llvm.ptr
// CHECK-NEXT:     %2 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 32 : i64}> : () -> i64
// CHECK-NEXT:     neura.br %2, %0, %3, %1, %4, %5 : i64, !llvm.ptr, i32, !llvm.ptr, i64, i64 to ^bb2
// CHECK-NEXT:   ^bb1:  // pred: ^bb4
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   ^bb2(%6: i64, %7: !llvm.ptr, %8: i32, %9: !llvm.ptr, %10: i64, %11: i64):  // 2 preds: ^bb0, ^bb4
// CHECK-NEXT:     %12 = "neura.gep"(%7, %6) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %13 = "neura.load"(%12) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %14 = "neura.icmp"(%13, %8) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CHECK-NEXT:     neura.cond_br %14 : i1 then %9, %6, %13, %10, %11, %7, %8 : !llvm.ptr, i64, i32, i64, i64, !llvm.ptr, i32 to ^bb3 else %10, %11, %7, %8, %9 : i64, i64, !llvm.ptr, i32, !llvm.ptr to ^bb4
// CHECK-NEXT:   ^bb3(%15: !llvm.ptr, %16: i64, %17: i32, %18: i64, %19: i64, %20: !llvm.ptr, %21: i32):  // pred: ^bb2
// CHECK-NEXT:     %22 = "neura.gep"(%15, %16) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %23 = "neura.load"(%22) : (!llvm.ptr) -> i32
// CHECK-NEXT:     %24 = "neura.add"(%23, %17) : (i32, i32) -> i32
// CHECK-NEXT:     "neura.store"(%24, %22) : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:     neura.br %18, %19, %20, %21, %15 : i64, i64, !llvm.ptr, i32, !llvm.ptr to ^bb4
// CHECK-NEXT:   ^bb4(%25: i64, %26: i64, %27: !llvm.ptr, %28: i32, %29: !llvm.ptr):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:     %30 = "neura.add"(%6, %25) : (i64, i64) -> i64
// CHECK-NEXT:     %31 = "neura.icmp"(%30, %26) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT:     neura.cond_br %31 : i1 then to ^bb1 else %30, %27, %28, %29, %25, %26 : i64, !llvm.ptr, i32, !llvm.ptr, i64, i64 to ^bb2
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK-NEXT: }


// CTRL2DATA:      module attributes {{.*}}
// CTRL2DATA-NEXT:   llvm.mlir.global external local_unnamed_addr @input(dense<[1, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31]> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
// CTRL2DATA-NEXT:   llvm.mlir.global external local_unnamed_addr @output(dense<0> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
// CTRL2DATA-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("output[%d] = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CTRL2DATA-NEXT:   llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CTRL2DATA-NEXT:     %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// CTRL2DATA-NEXT:     %1 = llvm.mlir.addressof @input : !llvm.ptr
// CTRL2DATA-NEXT:     %2 = llvm.mlir.addressof @output : !llvm.ptr
// CTRL2DATA-NEXT:     %3 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// CTRL2DATA-NEXT:     %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CTRL2DATA-NEXT:     %7 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 32 : i64}> : () -> i64
// CTRL2DATA-NEXT:     "neura.memset"(%2, %3, %4) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CTRL2DATA-NEXT:     neura.br %5 : i64 to ^bb1
// CTRL2DATA-NEXT:   ^bb1(%9: i64):  // 2 preds: ^bb0, ^bb3
// CTRL2DATA-NEXT:     %10 = "neura.gep"(%1, %9) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CTRL2DATA-NEXT:     %11 = "neura.load"(%10) : (!llvm.ptr) -> i32
// CTRL2DATA-NEXT:     %12 = "neura.icmp"(%11, %6) <{cmpType = "sgt"}> : (i32, i32) -> i1
// CTRL2DATA-NEXT:     neura.cond_br %12 : i1 then to ^bb2 else to ^bb3
// CTRL2DATA-NEXT:   ^bb2:  // pred: ^bb1
// CTRL2DATA-NEXT:     %13 = "neura.gep"(%2, %9) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CTRL2DATA-NEXT:     %14 = "neura.load"(%13) : (!llvm.ptr) -> i32
// CTRL2DATA-NEXT:     %15 = "neura.add"(%14, %11) : (i32, i32) -> i32
// CTRL2DATA-NEXT:     "neura.store"(%15, %13) : (i32, !llvm.ptr) -> ()
// CTRL2DATA-NEXT:     neura.br to ^bb3
// CTRL2DATA-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
// CTRL2DATA-NEXT:     %16 = "neura.add"(%9, %7) : (i64, i64) -> i64
// CTRL2DATA-NEXT:     %17 = "neura.icmp"(%16, %8) <{cmpType = "eq"}> : (i64, i64) -> i1
// CTRL2DATA-NEXT:     neura.cond_br %17 : i1 then %5 : i64 to ^bb5 else %16 : i64 to ^bb1
// CTRL2DATA-NEXT:   ^bb4:  // pred: ^bb5
// CTRL2DATA-NEXT:     "neura.return"(%6) : (i32) -> ()
// CTRL2DATA-NEXT:   ^bb5(%18: i64):  // 2 preds: ^bb3, ^bb5
// CTRL2DATA-NEXT:     %19 = "neura.constant"() <{value = 0 : i32}> : () -> index
// CTRL2DATA-NEXT:     %20 = "neura.gep"(%2, %19, %18) <{operandSegmentSizes = array<i32: 1, 2>}> : (!llvm.ptr, index, i64) -> !llvm.ptr
// CTRL2DATA-NEXT:     %21 = "neura.load"(%20) : (!llvm.ptr) -> i32
// CTRL2DATA-NEXT:     %22 = "neura.cast"(%18) <{cast_type = "trunc"}> : (i64) -> i32
// CTRL2DATA-NEXT:     %23 = llvm.call tail @printf(%0, %22, %21) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32) -> i32
// CTRL2DATA-NEXT:     %24 = "neura.add"(%18, %7) : (i64, i64) -> i64
// CTRL2DATA-NEXT:     %25 = "neura.icmp"(%24, %8) <{cmpType = "eq"}> : (i64, i64) -> i1
// CTRL2DATA-NEXT:     neura.cond_br %25 : i1 then to ^bb4 else %24 : i64 to ^bb5
// CTRL2DATA-NEXT:   }
// CTRL2DATA-NEXT:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CTRL2DATA-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %4 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %6 = "neura.constant"() <{value = 0 : i32}> : () -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %8 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %10 = "neura.constant"() <{value = 32 : i64}> : () -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %12 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %13 = neura.phi_start %11, %12 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %15 = neura.phi_start %9, %14 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %16 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %17 = neura.phi_start %3, %16 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %18 = neura.reserve : !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %19 = neura.phi_start %7, %18 : !neura.data<i32, i1>, !neura.data<i32, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %21 = neura.phi_start %1, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %22 = neura.reserve : !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %23 = neura.phi_start %5, %22 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %24 = "neura.gep"(%21, %23) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %25 = "neura.load"(%24) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %26 = "neura.icmp"(%25, %19) <{cmpType = "sgt"}> : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %27 = neura.grant_predicate %17, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %28 = neura.grant_predicate %23, %26 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %29 = neura.grant_predicate %25, %26 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %30 = neura.grant_predicate %15, %26 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %31 = neura.grant_predicate %13, %26 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %32 = neura.grant_predicate %21, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %33 = neura.grant_predicate %19, %26 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %34 = "neura.not"(%26) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %35 = neura.grant_predicate %15, %34 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %36 = neura.grant_predicate %13, %34 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %37 = neura.grant_predicate %21, %34 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %38 = neura.grant_predicate %19, %34 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %39 = neura.grant_predicate %17, %34 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %40 = "neura.gep"(%27, %28) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %41 = "neura.load"(%40) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %42 = "neura.add"(%41, %29) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     "neura.store"(%42, %40) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CTRL2DATA-NEXT:     %43 = "neura.phi"(%39, %27) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %44 = "neura.phi"(%38, %33) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %45 = "neura.phi"(%37, %32) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %46 = "neura.phi"(%36, %31) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %47 = "neura.phi"(%35, %30) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %48 = "neura.add"(%23, %47) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %49 = "neura.icmp"(%48, %46) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %50 = "neura.not"(%49) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %51 = neura.grant_predicate %48, %50 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %51 -> %22 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %52 = neura.grant_predicate %45, %50 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %52 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %53 = neura.grant_predicate %44, %50 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %53 -> %18 : !neura.data<i32, i1> !neura.data<i32, i1>
// CTRL2DATA-NEXT:     %54 = neura.grant_predicate %43, %50 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %54 -> %16 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CTRL2DATA-NEXT:     %55 = neura.grant_predicate %47, %50 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %55 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %56 = neura.grant_predicate %46, %50 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     neura.ctrl_mov %56 -> %12 : !neura.data<i64, i1> !neura.data<i64, i1>
// CTRL2DATA-NEXT:     %57 = "neura.constant"() <{value = true}> : () -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     %58 = "neura.grant_once"(%57) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CTRL2DATA-NEXT:     "neura.return"(%58) : (!neura.data<i1, i1>) -> ()
// CTRL2DATA-NEXT:   }
// CTRL2DATA-NEXT:   llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CTRL2DATA-NEXT: }


// MAPPING:      [DEBUG] Recurrence cycle (length 3):
// MAPPING-NEXT:   %1 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:   %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   %25 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %31 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   neura.ctrl_mov %33 -> %1 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: [DEBUG] Recurrence cycle (length 5):
// MAPPING-NEXT:   %1 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:   %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   %25 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %29 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %32 = "neura.data_mov"(%30) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   neura.ctrl_mov %33 -> %1 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Longest recurrence cycle (length 5):
// MAPPING-NEXT: %1 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: %25 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %29 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %32 = "neura.data_mov"(%30) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: neura.ctrl_mov %33 -> %1 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %1 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %34 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %2 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %35 = "neura.data_mov"(%34) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: "neura.return"(%35) : (!neura.data<i1, i1>) -> ()
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %25 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %31 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %6 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %7 = "neura.load"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %29 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %13 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %8 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %32 = "neura.data_mov"(%30) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %14 = "neura.data_mov"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %11 = "neura.data_mov"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %15 = neura.grant_predicate %13, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %33 -> %1 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %21 = "neura.data_mov"(%15) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %16 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %24 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %18 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %20 = "neura.data_mov"(%19) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %22 = "neura.add"(%20, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: %23 = "neura.data_mov"(%22) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] Topologically sorted op: "neura.store"(%23, %24) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 0: 3 ops
// MAPPING-NEXT:   %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT:   %1 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:   %2 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 1: 3 ops
// MAPPING-NEXT:   %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   %25 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 2: 5 ops
// MAPPING-NEXT:   %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:   %31 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %6 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 3: 4 ops
// MAPPING-NEXT:   %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %7 = "neura.load"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %29 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %8 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 4: 5 ops
// MAPPING-NEXT:   %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:   %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %32 = "neura.data_mov"(%30) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %11 = "neura.data_mov"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 5: 4 ops
// MAPPING-NEXT:   %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:   neura.ctrl_mov %33 -> %1 : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:   %16 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 6: 4 ops
// MAPPING-NEXT:   %13 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %14 = "neura.data_mov"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:   %18 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 7: 4 ops
// MAPPING-NEXT:   %15 = neura.grant_predicate %13, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:   %21 = "neura.data_mov"(%15) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %20 = "neura.data_mov"(%19) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 8: 5 ops
// MAPPING-NEXT:   %34 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// MAPPING-NEXT:   %35 = "neura.data_mov"(%34) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:   %24 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:   %22 = "neura.add"(%20, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:   %23 = "neura.data_mov"(%22) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP Bucket Level 9: 2 ops
// MAPPING-NEXT:   "neura.return"(%35) : (!neura.data<i1, i1>) -> ()
// MAPPING-NEXT:   "neura.store"(%23, %24) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 0)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %1 = neura.reserve : !neura.data<i64, i1> (ALAP level: 0)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %2 = "neura.data_mov"(%0) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 0)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 1)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %25 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %4 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %31 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %27 = "neura.data_mov"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %6 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 2)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1> (ALAP level: 3)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %7 = "neura.load"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 3)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %29 = "neura.data_mov"(%28) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 3)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %8 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 3)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %10 = "neura.data_mov"(%3) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 4)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %32 = "neura.data_mov"(%30) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %11 = "neura.data_mov"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %33 -> %1 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 5)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %16 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %13 = "neura.data_mov"(%7) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 6)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %14 = "neura.data_mov"(%9) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 6)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %18 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %15 = neura.grant_predicate %13, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1> (ALAP level: 7)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %21 = "neura.data_mov"(%15) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %20 = "neura.data_mov"(%19) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %34 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1> (ALAP level: 8)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %35 = "neura.data_mov"(%34) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %24 = "neura.data_mov"(%17) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %22 = "neura.add"(%20, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 8)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: %23 = "neura.data_mov"(%22) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 8)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: "neura.return"(%35) : (!neura.data<i1, i1>) -> () (ALAP level: 9)
// MAPPING-NEXT: [MapToAcceleratorPass] ALAP sorted op: "neura.store"(%23, %24) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (ALAP level: 9)
// MAPPING-NEXT: ---------------------------------------------------------
// MAPPING-NEXT: [HeuristicMapping] Starting mapping with 39 operations.
// MAPPING-NEXT: Configuration: MAX Backtrack Depth = 3, MAX Candidate Locations = 5
// MAPPING-NEXT: [HeuristicMapping] Filtered 22 non-materialized operations, 17 operations require physical mapping.
// MAPPING-NEXT: [HeuristicMapping] Materialized operations list:
// MAPPING-NEXT: 0 %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
// MAPPING-NEXT: 1 %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
// MAPPING-NEXT: 2 %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 2)
// MAPPING-NEXT: 3 %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 2)
// MAPPING-NEXT: 4 %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 3)
// MAPPING-NEXT: 5 %7 = "neura.load"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 3)
// MAPPING-NEXT: 6 %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 4)
// MAPPING-NEXT: 7 %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1> (level: 4)
// MAPPING-NEXT: 8 %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
// MAPPING-NEXT: 9 %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
// MAPPING-NEXT: 10 %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
// MAPPING-NEXT: 11 %15 = neura.grant_predicate %13, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1> (level: 7)
// MAPPING-NEXT: 12 %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
// MAPPING-NEXT: 13 %34 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1> (level: 8)
// MAPPING-NEXT: 14 %22 = "neura.add"(%20, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
// MAPPING-NEXT: 15 "neura.return"(%35) : (!neura.data<i1, i1>) -> () (level: 9)
// MAPPING-NEXT: 16 "neura.store"(%23, %24) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
// MAPPING-NEXT: [HeuristicMapping] Found 80 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#11 @t=0
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 54 candidate locations for operation: %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#11 @t=1
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=0 to Tile#11 @t=1
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #704
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %3 = neura.phi_start %2, %1 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 53 candidate locations for operation: %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#11 @t=2
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=1 to Tile#11 @t=2
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #704
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %26 = "neura.add"(%25) {rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 52 candidate locations for operation: %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#10 @t=2
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=1 to Tile#10 @t=2
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg0"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 51 candidate locations for operation: %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#11 @t=3
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=2 to Tile#11 @t=3
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #704
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 58 candidate locations for operation: %7 = "neura.load"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#10 @t=3
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#10 @t=3
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #640
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %7 = "neura.load"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 49 candidate locations for operation: %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#10 @t=4
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=3 to Tile#10 @t=4
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 56 candidate locations for operation: %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#11 @t=4
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#11 @t=4
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 1 candidate locations for operation: %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=2 to Tile#10 @t=5
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#10 @t=5
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #641
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#11 @t=6
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %33 = neura.grant_predicate %31, %32 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 46 candidate locations for operation: %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#7 @t=5
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=1 to Tile#7 @t=5
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=4 to Tile#7 @t=5
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %12 = neura.grant_predicate %10, %11 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 45 candidate locations for operation: %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#7 @t=6
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#7 @t=5 to Tile#7 @t=6
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #448
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {lhs_value = "%arg1"} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 64 candidate locations for operation: %15 = neura.grant_predicate %13, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#15 @t=7
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#15 @t=7
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#11 @t=4 to Tile#15 @t=7
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %15 = neura.grant_predicate %13, %14 : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 45 candidate locations for operation: %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#7 @t=7
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#7 @t=6 to Tile#7 @t=7
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #448
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 67 candidate locations for operation: %34 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#7 @t=8
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %34 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 28 candidate locations for operation: %22 = "neura.add"(%20, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#7 @t=9
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#7 @t=7 to Tile#7 @t=9
// MAPPING-NEXT: [tryRouteDataMove] Successfully routed on same tile using Register #449
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#15 @t=7 to Tile#7 @t=9
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation %22 = "neura.add"(%20, %21) : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT: [HeuristicMapping] Found 40 candidate locations for operation: "neura.return"(%35) : (!neura.data<i1, i1>) -> ()
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#3 @t=9
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#7 @t=8 to Tile#3 @t=9
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation "neura.return"(%35) : (!neura.data<i1, i1>) -> ()
// MAPPING-NEXT: [HeuristicMapping] Found 40 candidate locations for operation: "neura.store"(%23, %24) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: [HeuristicMapping] Trying candidate 1/5 at tile#6 @t=10
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#7 @t=9 to Tile#6 @t=10
// MAPPING-NEXT: [tryRouteDataMove] Routing from Tile#7 @t=6 to Tile#6 @t=10
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped operation "neura.store"(%23, %24) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT: [HeuristicMapping] Successfully mapped all 17 operations.
// MAPPING-NEXT: module attributes {{.*}}
// MAPPING-NEXT:   llvm.mlir.global external local_unnamed_addr @input(dense<[1, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23, 24, -25, 26, -27, 28, -29, 30, -31]> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
// MAPPING-NEXT:   llvm.mlir.global external local_unnamed_addr @output(dense<0> : tensor<32xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x i32>
// MAPPING-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("output[%d] = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// MAPPING-NEXT:   llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// MAPPING-NEXT:     %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// MAPPING-NEXT:     %1 = llvm.mlir.addressof @input : !llvm.ptr
// MAPPING-NEXT:     %2 = llvm.mlir.addressof @output : !llvm.ptr
// MAPPING-NEXT:     %3 = "neura.constant"() <{value = 0 : i8}> : () -> i8
// MAPPING-NEXT:     %4 = "neura.constant"() <{value = 128 : i64}> : () -> i64
// MAPPING-NEXT:     %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// MAPPING-NEXT:     %6 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// MAPPING-NEXT:     %7 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %8 = "neura.data_mov"(%3) : (i8) -> i8
// MAPPING-NEXT:     %9 = "neura.data_mov"(%4) : (i64) -> i64
// MAPPING-NEXT:     "neura.memset"(%7, %8, %9) <{is_volatile = false}> : (!llvm.ptr, i8, i64) -> ()
// MAPPING-NEXT:     %10 = "neura.data_mov"(%5) : (i64) -> i64
// MAPPING-NEXT:     neura.br %10 : i64 to ^bb1
// MAPPING-NEXT:   ^bb1(%11: i64):  // 2 preds: ^bb0, ^bb3
// MAPPING-NEXT:     %12 = "neura.data_mov"(%1) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %13 = "neura.data_mov"(%11) : (i64) -> i64
// MAPPING-NEXT:     %14 = "neura.gep"(%12, %13) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// MAPPING-NEXT:     %15 = "neura.data_mov"(%14) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %16 = "neura.load"(%15) : (!llvm.ptr) -> i32
// MAPPING-NEXT:     %17 = "neura.data_mov"(%16) : (i32) -> i32
// MAPPING-NEXT:     %18 = "neura.icmp"(%17) <{cmpType = "sgt"}> {rhs_value = 0 : i32} : (i32) -> i1
// MAPPING-NEXT:     %19 = "neura.data_mov"(%18) : (i1) -> i1
// MAPPING-NEXT:     neura.cond_br %19 : i1 then to ^bb2 else to ^bb3
// MAPPING-NEXT:   ^bb2:  // pred: ^bb1
// MAPPING-NEXT:     %20 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %21 = "neura.data_mov"(%11) : (i64) -> i64
// MAPPING-NEXT:     %22 = "neura.gep"(%20, %21) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// MAPPING-NEXT:     %23 = "neura.data_mov"(%22) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %24 = "neura.load"(%23) : (!llvm.ptr) -> i32
// MAPPING-NEXT:     %25 = "neura.data_mov"(%24) : (i32) -> i32
// MAPPING-NEXT:     %26 = "neura.data_mov"(%16) : (i32) -> i32
// MAPPING-NEXT:     %27 = "neura.add"(%25, %26) : (i32, i32) -> i32
// MAPPING-NEXT:     %28 = "neura.data_mov"(%27) : (i32) -> i32
// MAPPING-NEXT:     %29 = "neura.data_mov"(%22) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     "neura.store"(%28, %29) : (i32, !llvm.ptr) -> ()
// MAPPING-NEXT:     neura.br to ^bb3
// MAPPING-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
// MAPPING-NEXT:     %30 = "neura.data_mov"(%11) : (i64) -> i64
// MAPPING-NEXT:     %31 = "neura.add"(%30) {rhs_value = 1 : i64} : (i64) -> i64
// MAPPING-NEXT:     %32 = "neura.data_mov"(%31) : (i64) -> i64
// MAPPING-NEXT:     %33 = "neura.icmp"(%32) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (i64) -> i1
// MAPPING-NEXT:     %34 = "neura.data_mov"(%33) : (i1) -> i1
// MAPPING-NEXT:     %35 = "neura.data_mov"(%5) : (i64) -> i64
// MAPPING-NEXT:     %36 = "neura.data_mov"(%31) : (i64) -> i64
// MAPPING-NEXT:     neura.cond_br %34 : i1 then %35 : i64 to ^bb5 else %36 : i64 to ^bb1
// MAPPING-NEXT:   ^bb4:  // pred: ^bb5
// MAPPING-NEXT:     %37 = "neura.data_mov"(%6) : (i32) -> i32
// MAPPING-NEXT:     "neura.return"(%37) : (i32) -> ()
// MAPPING-NEXT:   ^bb5(%38: i64):  // 2 preds: ^bb3, ^bb5
// MAPPING-NEXT:     %39 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) : (i64) -> i64
// MAPPING-NEXT:     %41 = "neura.gep"(%39, %40) <{operandSegmentSizes = array<i32: 1, 1>}> {operand_1_value = 0 : i32} : (!llvm.ptr, i64) -> !llvm.ptr
// MAPPING-NEXT:     %42 = "neura.data_mov"(%41) : (!llvm.ptr) -> !llvm.ptr
// MAPPING-NEXT:     %43 = "neura.load"(%42) : (!llvm.ptr) -> i32
// MAPPING-NEXT:     %44 = "neura.data_mov"(%38) : (i64) -> i64
// MAPPING-NEXT:     %45 = "neura.cast"(%44) <{cast_type = "trunc"}> : (i64) -> i32
// MAPPING-NEXT:     %46 = llvm.call tail @printf(%0, %45, %43) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, i32, i32) -> i32
// MAPPING-NEXT:     %47 = "neura.data_mov"(%38) : (i64) -> i64
// MAPPING-NEXT:     %48 = "neura.add"(%47) {rhs_value = 1 : i64} : (i64) -> i64
// MAPPING-NEXT:     %49 = "neura.data_mov"(%48) : (i64) -> i64
// MAPPING-NEXT:     %50 = "neura.icmp"(%49) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (i64) -> i1
// MAPPING-NEXT:     %51 = "neura.data_mov"(%50) : (i1) -> i1
// MAPPING-NEXT:     %52 = "neura.data_mov"(%48) : (i64) -> i64
// MAPPING-NEXT:     neura.cond_br %51 : i1 then to ^bb4 else %52 : i64 to ^bb5
// MAPPING-NEXT:   }
// MAPPING-NEXT:   func.func @_Z6kernelPiS_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 5 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 5 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// MAPPING-NEXT:     %0 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 0 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 3 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = neura.reserve {dfg_id = 1 : i32} : !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.data_mov"(%0) {dfg_id = 3 : i32, mapping_locs = [{id = 704 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = neura.phi_start %2, %1 {dfg_id = 5 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 3 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%3) {dfg_id = 9 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.gep"(%4) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 11 : i32, lhs_value = "%arg0", mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %6 = "neura.data_mov"(%5) {dfg_id = 14 : i32, mapping_locs = [{id = 640 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %7 = "neura.load"(%6) {dfg_id = 16 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %8 = "neura.data_mov"(%7) {dfg_id = 19 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %9 = "neura.icmp"(%8) <{cmpType = "sgt"}> {dfg_id = 21 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 0 : i32} : (!neura.data<i32, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%3) {dfg_id = 8 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 448 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 448 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 448 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %11 = "neura.data_mov"(%9) {dfg_id = 24 : i32, mapping_locs = [{id = 36 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %12 = neura.grant_predicate %10, %11 {dfg_id = 27 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 3 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%7) {dfg_id = 18 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 44 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 960 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 960 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %14 = "neura.data_mov"(%9) {dfg_id = 23 : i32, mapping_locs = [{id = 37 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 961 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 961 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %15 = neura.grant_predicate %13, %14 {dfg_id = 26 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 3 : i32}]} : !neura.data<i32, i1>, !neura.data<i1, i1> -> !neura.data<i32, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%12) {dfg_id = 30 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.gep"(%16) <{operandSegmentSizes = array<i32: 0, 1>}> {dfg_id = 31 : i32, lhs_value = "%arg1", mapping_locs = [{id = 7 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 6 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %18 = "neura.data_mov"(%17) {dfg_id = 33 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     %19 = "neura.load"(%18) {dfg_id = 34 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %20 = "neura.data_mov"(%19) {dfg_id = 35 : i32, mapping_locs = [{id = 449 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 449 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %21 = "neura.data_mov"(%15) {dfg_id = 29 : i32, mapping_locs = [{id = 47 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 7 : i32}, {id = 36 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %22 = "neura.add"(%20, %21) {dfg_id = 36 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 3 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %23 = "neura.data_mov"(%22) {dfg_id = 37 : i32, mapping_locs = [{id = 21 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
// MAPPING-NEXT:     %24 = "neura.data_mov"(%17) {dfg_id = 32 : i32, mapping_locs = [{id = 21 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 6 : i32}, {id = 384 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 384 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 384 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MAPPING-NEXT:     "neura.store"(%23, %24) {dfg_id = 38 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 2 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MAPPING-NEXT:     %25 = "neura.data_mov"(%3) {dfg_id = 7 : i32, mapping_locs = [{id = 704 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %26 = "neura.add"(%25) {dfg_id = 10 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 1 : i64} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %27 = "neura.data_mov"(%26) {dfg_id = 13 : i32, mapping_locs = [{id = 704 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %28 = "neura.icmp"(%27) <{cmpType = "eq"}> {dfg_id = 15 : i32, mapping_locs = [{id = 11 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 2 : i32}], rhs_value = 32 : i64} : (!neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %29 = "neura.data_mov"(%28) {dfg_id = 17 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %30 = "neura.not"(%29) {dfg_id = 20 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 2 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%26) {dfg_id = 12 : i32, mapping_locs = [{id = 35 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 640 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 640 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %32 = "neura.data_mov"(%30) {dfg_id = 22 : i32, mapping_locs = [{id = 641 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %33 = neura.grant_predicate %31, %32 {dfg_id = 25 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %33 -> %1 {dfg_id = 28 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 5 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %34 = "neura.grant_once"() <{constant_value = true}> {dfg_id = 2 : i32, mapping_locs = [{id = 7 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 8 : i32, x = 3 : i32, y = 1 : i32}]} : () -> !neura.data<i1, i1>
// MAPPING-NEXT:     %35 = "neura.data_mov"(%34) {dfg_id = 4 : i32, mapping_locs = [{id = 22 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     "neura.return"(%35) {dfg_id = 6 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 9 : i32, x = 3 : i32, y = 0 : i32}]} : (!neura.data<i1, i1>) -> ()
// MAPPING-NEXT:   }
// MAPPING-NEXT:   llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// MAPPING-NEXT: }