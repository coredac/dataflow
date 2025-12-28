// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O1 -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// TODO: Enable --insert-mov once the backward ctrl flow mov is supported.
// Lowers to neura.
// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:  | FileCheck %s

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --fuse-pattern \
// RUN:  | FileCheck %s --check-prefix=CHECK-FUSED

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --fuse-pattern \
// RUN:   --insert-data-mov \
// RUN:  | FileCheck %s --check-prefix=CHECK-MOV

// CHECK:      module attributes {{.*}}
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @input(dense<1.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @output(dense<0.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-NEXT:   llvm.mlir.global external local_unnamed_addr @coefficients(dense<[2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00]> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("output: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-NEXT:   llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:     %1 = llvm.mlir.addressof @coefficients : !llvm.ptr
// CHECK-NEXT:     %2 = llvm.mlir.addressof @input : !llvm.ptr
// CHECK-NEXT:     %3 = llvm.mlir.addressof @output : !llvm.ptr
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:     %6 = "neura.constant"() <{value = 32 : i64}> : () -> i64
// CHECK-NEXT:     %7 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:     %8 = "neura.load"(%3) : (!llvm.ptr) -> f32
// CHECK-NEXT:     neura.br %4, %8 : i64, f32 to ^bb1
// CHECK-NEXT:   ^bb1(%9: i64, %10: f32):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:     %11 = "neura.gep"(%2, %9) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %12 = "neura.load"(%11) : (!llvm.ptr) -> f32
// CHECK-NEXT:     %13 = "neura.gep"(%1, %9) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT:     %14 = "neura.load"(%13) : (!llvm.ptr) -> f32
// CHECK-NEXT:     %15 = "neura.fmul"(%12, %14) : (f32, f32) -> f32
// CHECK-NEXT:     %16 = "neura.fadd"(%10, %15) : (f32, f32) -> f32
// CHECK-NEXT:     %17 = "neura.add"(%9, %5) : (i64, i64) -> i64
// CHECK-NEXT:     %18 = "neura.icmp"(%17, %6) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT:     neura.cond_br %18 : i1 then to ^bb2 else %17, %16 : i64, f32 to ^bb1
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     "neura.store"(%16, %3) : (f32, !llvm.ptr) -> ()
// CHECK-NEXT:     %19 = llvm.fpext %16 : f32 to f64
// CHECK-NEXT:     %20 = llvm.call tail @printf(%0, %19) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, f64) -> i32
// CHECK-NEXT:     "neura.return"(%7) : (i32) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CHECK-NEXT:     %0 = "neura.constant"() <{value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %1 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %2 = "neura.constant"() <{value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %3 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %4 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %5 = "neura.constant"() <{value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %6 = "neura.load"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     neura.br %3, %6, %0, %2, %1, %4, %5 : !neura.data<i64, i1>, !neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> to ^bb1
// CHECK-NEXT:   ^bb1(%7: !neura.data<i64, i1>, %8: !neura.data<f32, i1>, %9: !neura.data<!llvm.ptr, i1>, %10: !neura.data<!llvm.ptr, i1>, %11: !neura.data<!llvm.ptr, i1>, %12: !neura.data<i64, i1>, %13: !neura.data<i64, i1>):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:     %14 = "neura.gep"(%9, %7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %15 = "neura.load"(%14) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     %16 = "neura.gep"(%10, %7) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %17 = "neura.load"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     %18 = "neura.fmul"(%15, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     %19 = "neura.fadd"(%8, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     "neura.store"(%19, %11) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-NEXT:     %20 = "neura.add"(%7, %12) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %21 = "neura.icmp"(%20, %13) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     neura.cond_br %21 : !neura.data<i1, i1> then to ^bb2 else %20, %19, %9, %10, %11, %12, %13 : !neura.data<i64, i1>, !neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> to ^bb1
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK-NEXT: }

// Verifies the neura ops are generated. And fusion happens.
// CHECK-FUSED:      module attributes {{.*}}
// CHECK-FUSED-NEXT:   llvm.mlir.global external local_unnamed_addr @input(dense<1.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-FUSED-NEXT:   llvm.mlir.global external local_unnamed_addr @output(dense<0.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-FUSED-NEXT:   llvm.mlir.global external local_unnamed_addr @coefficients(dense<[2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00]> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-FUSED-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("output: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-FUSED-NEXT:   llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-FUSED-NEXT:     %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-FUSED-NEXT:     %1 = llvm.mlir.addressof @coefficients : !llvm.ptr
// CHECK-FUSED-NEXT:     %2 = llvm.mlir.addressof @input : !llvm.ptr
// CHECK-FUSED-NEXT:     %3 = llvm.mlir.addressof @output : !llvm.ptr
// CHECK-FUSED-NEXT:     %4 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-FUSED-NEXT:     %5 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-FUSED-NEXT:     %6 = "neura.load"(%3) : (!llvm.ptr) -> f32
// CHECK-FUSED-NEXT:     neura.br %4, %6 : i64, f32 to ^bb1
// CHECK-FUSED-NEXT:   ^bb1(%7: i64, %8: f32):  // 2 preds: ^bb0, ^bb1
// CHECK-FUSED-NEXT:     %9 = neura.load_indexed %2[%7 : i64] !llvm.ptr : f32
// CHECK-FUSED-NEXT:     %10 = neura.load_indexed %1[%7 : i64] !llvm.ptr : f32
// CHECK-FUSED-NEXT:     %11 = "neura.fmul_fadd"(%9, %10, %8) : (f32, f32, f32) -> f32
// CHECK-FUSED-NEXT:     %12 = "neura.add"(%7) {rhs_value = 1 : i64} : (i64) -> i64
// CHECK-FUSED-NEXT:     %13 = "neura.icmp"(%12) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (i64) -> i1
// CHECK-FUSED-NEXT:     neura.cond_br %13 : i1 then to ^bb2 else %12, %11 : i64, f32 to ^bb1
// CHECK-FUSED-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-FUSED-NEXT:     "neura.store"(%11, %3) : (f32, !llvm.ptr) -> ()
// CHECK-FUSED-NEXT:     %14 = llvm.fpext %11 : f32 to f64
// CHECK-FUSED-NEXT:     %15 = llvm.call tail @printf(%0, %14) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, f64) -> i32
// CHECK-FUSED-NEXT:     "neura.return"(%5) : (i32) -> ()
// CHECK-FUSED-NEXT:   }
// CHECK-FUSED-NEXT:   func.func @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CHECK-FUSED-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %3 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %5 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %6 = "neura.grant_once"() <{constant_value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %7 = "neura.load"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %8 = "neura.grant_once"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %9 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %10 = neura.phi_start %6, %9 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %11 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %12 = neura.phi_start %5, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %13 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %14 = neura.phi_start %1, %13 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %15 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %16 = neura.phi_start %3, %15 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %18 = neura.phi_start %0, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %19 = neura.reserve : !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %20 = neura.phi_start %8, %19 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %22 = neura.phi_start %4, %21 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %23 = neura.load_indexed %18[%22 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %24 = neura.load_indexed %16[%22 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %25 = "neura.fmul_fadd"(%23, %24, %20) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     "neura.store"(%25, %14) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-FUSED-NEXT:     %26 = "neura.add"(%22, %12) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %27 = "neura.icmp"(%26, %10) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     %28 = "neura.not"(%27) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     %29 = neura.grant_predicate %26, %28 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %29 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %30 = neura.grant_predicate %25, %28 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %30 -> %19 : !neura.data<f32, i1> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %31 = neura.grant_predicate %18, %28 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %31 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %32 = neura.grant_predicate %16, %28 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %32 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %33 = neura.grant_predicate %14, %28 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %33 -> %13 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %34 = neura.grant_predicate %12, %28 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %34 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %35 = neura.grant_predicate %10, %28 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %35 -> %9 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %36 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     "neura.return"(%36) : (!neura.data<i1, i1>) -> ()
// CHECK-FUSED-NEXT:   }
// CHECK-FUSED-NEXT:   llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK-FUSED-NEXT: }

// CHECK-MOV:      module attributes {{.*}}
// CHECK-MOV-NEXT:   llvm.mlir.global external local_unnamed_addr @input(dense<1.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-MOV-NEXT:   llvm.mlir.global external local_unnamed_addr @output(dense<0.000000e+00> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-MOV-NEXT:   llvm.mlir.global external local_unnamed_addr @coefficients(dense<[2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00, 2.500000e-01, 1.500000e+00, 3.750000e+00, -2.250000e+00, 5.000000e-01, 7.500000e-01, -3.000000e+00, 1.250000e+00]> : tensor<32xf32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<32 x f32>
// CHECK-MOV-NEXT:   llvm.mlir.global private unnamed_addr constant @".str"("output: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
// CHECK-MOV-NEXT:   llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-MOV-NEXT:     %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-MOV-NEXT:     %1 = llvm.mlir.addressof @coefficients : !llvm.ptr
// CHECK-MOV-NEXT:     %2 = llvm.mlir.addressof @input : !llvm.ptr
// CHECK-MOV-NEXT:     %3 = llvm.mlir.addressof @output : !llvm.ptr
// CHECK-MOV-NEXT:     %4 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-MOV-NEXT:     %5 = "neura.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-MOV-NEXT:     %6 = "neura.data_mov"(%3) : (!llvm.ptr) -> !llvm.ptr
// CHECK-MOV-NEXT:     %7 = "neura.load"(%6) : (!llvm.ptr) -> f32
// CHECK-MOV-NEXT:     %8 = "neura.data_mov"(%4) : (i64) -> i64
// CHECK-MOV-NEXT:     %9 = "neura.data_mov"(%7) : (f32) -> f32
// CHECK-MOV-NEXT:     neura.br %8, %9 : i64, f32 to ^bb1
// CHECK-MOV-NEXT:   ^bb1(%10: i64, %11: f32):  // 2 preds: ^bb0, ^bb1
// CHECK-MOV-NEXT:     %12 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
// CHECK-MOV-NEXT:     %13 = "neura.data_mov"(%10) : (i64) -> i64
// CHECK-MOV-NEXT:     %14 = neura.load_indexed %12[%13 : i64] !llvm.ptr : f32
// CHECK-MOV-NEXT:     %15 = "neura.data_mov"(%1) : (!llvm.ptr) -> !llvm.ptr
// CHECK-MOV-NEXT:     %16 = "neura.data_mov"(%10) : (i64) -> i64
// CHECK-MOV-NEXT:     %17 = neura.load_indexed %15[%16 : i64] !llvm.ptr : f32
// CHECK-MOV-NEXT:     %18 = "neura.data_mov"(%14) : (f32) -> f32
// CHECK-MOV-NEXT:     %19 = "neura.data_mov"(%17) : (f32) -> f32
// CHECK-MOV-NEXT:     %20 = "neura.data_mov"(%11) : (f32) -> f32
// CHECK-MOV-NEXT:     %21 = "neura.fmul_fadd"(%18, %19, %20) : (f32, f32, f32) -> f32
// CHECK-MOV-NEXT:     %22 = "neura.data_mov"(%10) : (i64) -> i64
// CHECK-MOV-NEXT:     %23 = "neura.add"(%22) {rhs_value = 1 : i64} : (i64) -> i64
// CHECK-MOV-NEXT:     %24 = "neura.data_mov"(%23) : (i64) -> i64
// CHECK-MOV-NEXT:     %25 = "neura.icmp"(%24) <{cmpType = "eq"}> {rhs_value = 32 : i64} : (i64) -> i1
// CHECK-MOV-NEXT:     %26 = "neura.data_mov"(%25) : (i1) -> i1
// CHECK-MOV-NEXT:     %27 = "neura.data_mov"(%23) : (i64) -> i64
// CHECK-MOV-NEXT:     %28 = "neura.data_mov"(%21) : (f32) -> f32
// CHECK-MOV-NEXT:     neura.cond_br %26 : i1 then to ^bb2 else %27, %28 : i64, f32 to ^bb1
// CHECK-MOV-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-MOV-NEXT:     %29 = "neura.data_mov"(%21) : (f32) -> f32
// CHECK-MOV-NEXT:     %30 = "neura.data_mov"(%3) : (!llvm.ptr) -> !llvm.ptr
// CHECK-MOV-NEXT:     "neura.store"(%29, %30) : (f32, !llvm.ptr) -> ()
// CHECK-MOV-NEXT:     %31 = llvm.fpext %21 : f32 to f64
// CHECK-MOV-NEXT:     %32 = llvm.call tail @printf(%0, %31) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr, f64) -> i32
// CHECK-MOV-NEXT:     %33 = "neura.data_mov"(%5) : (i32) -> i32
// CHECK-MOV-NEXT:     "neura.return"(%33) : (i32) -> ()
// CHECK-MOV-NEXT:   }
// CHECK-MOV-NEXT:   func.func @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
// CHECK-MOV-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %2 = "neura.constant"() <{value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %3 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %5 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %6 = "neura.grant_once"() <{constant_value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %7 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %8 = "neura.load"(%7) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %9 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %10 = "neura.grant_once"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %11 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %12 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %15 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %16 = neura.phi_start %15, %14 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %18 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %21 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %23 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %24 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %25 = neura.phi_start %24, %23 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %26 = neura.reserve : !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %27 = "neura.data_mov"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %28 = neura.phi_start %27, %26 : !neura.data<f32, i1>, !neura.data<f32, i1> -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %29 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %30 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %31 = neura.phi_start %30, %29 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %32 = "neura.data_mov"(%25) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %33 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %34 = neura.load_indexed %32[%33 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %35 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %36 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %37 = neura.load_indexed %35[%36 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %38 = "neura.data_mov"(%34) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %39 = "neura.data_mov"(%37) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %40 = "neura.data_mov"(%28) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %41 = "neura.fmul_fadd"(%38, %39, %40) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %42 = "neura.data_mov"(%41) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %43 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     "neura.store"(%42, %43) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-MOV-NEXT:     %44 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %45 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %46 = "neura.add"(%44, %45) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %47 = "neura.data_mov"(%46) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %48 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %49 = "neura.icmp"(%47, %48) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %50 = "neura.data_mov"(%49) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %51 = "neura.not"(%50) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %52 = "neura.data_mov"(%46) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %53 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %54 = neura.grant_predicate %52, %53 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %54 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %55 = "neura.data_mov"(%41) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %56 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %57 = neura.grant_predicate %55, %56 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %57 -> %26 : !neura.data<f32, i1> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %58 = "neura.data_mov"(%25) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %59 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %60 = neura.grant_predicate %58, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %60 -> %23 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %61 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %62 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %63 = neura.grant_predicate %61, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %63 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %64 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %65 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %66 = neura.grant_predicate %64, %65 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %66 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %67 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %68 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %69 = neura.grant_predicate %67, %68 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %69 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %70 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %71 = "neura.data_mov"(%51) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %72 = neura.grant_predicate %70, %71 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %72 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %73 = "neura.grant_once"() <{constant_value = true}> : () -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %74 = "neura.data_mov"(%73) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     "neura.return"(%74) : (!neura.data<i1, i1>) -> ()
// CHECK-MOV-NEXT:   }
// CHECK-MOV-NEXT:   llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"}
// CHECK-MOV-NEXT: }