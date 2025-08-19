// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O1 -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// TODO: Enable --insert-mov once the backward ctrl flow mov is supported.
// Lowers to neura.
// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:  | FileCheck %s

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --fuse-pattern \
// RUN:  | FileCheck %s --check-prefix=FUSE

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --fuse-pattern \
// RUN:   --insert-data-mov \
// RUN:  | FileCheck %s --check-prefix=MOV

// CHECK: llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %1 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %4 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %5 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:     %6 = "neura.load"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     neura.br %3, %6, %0, %2, %1, %4, %5 : !neura.data<i64, i1>, !neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> to ^bb1
// CHECK-NEXT:   ^bb1(%7: !neura.data<i64, i1>, %8: !neura.data<f32, i1>, %9: !neura.data<!llvm.ptr, i1>, %10: !neura.data<!llvm.ptr, i1>, %11: !neura.data<!llvm.ptr, i1>, %12: !neura.data<i64, i1>, %13: !neura.data<i64, i1>):  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:     %14 = "neura.gep"(%9, %7) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %15 = "neura.load"(%14) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     %16 = "neura.gep"(%10, %7) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-NEXT:     %17 = "neura.load"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     %18 = "neura.fmul"(%15, %17) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     %19 = "neura.fadd"(%8, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:     "neura.store"(%19, %11) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-NEXT:     %20 = "neura.add"(%7, %12) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:     %21 = "neura.icmp"(%20, %13) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:     neura.cond_br %21 : !neura.data<i1, i1> then to ^bb2 else %20, %19, %0, %2, %1, %4, %5 : !neura.data<i64, i1>, !neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1> to ^bb1
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     "neura.return"() : () -> ()
// CHECK-NEXT:   }

// Verifies the neura ops are generated. And fusion happens.
// FUSE:        llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// FUSE-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %3 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %5 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %6 = "neura.grant_once"() <{constant_value = 32 : i64}> : () -> !neura.data<i64, i1>
// FUSE-NEXT:     %7 = "neura.load"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %8 = "neura.grant_once"(%7) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %9 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %10 = "neura.phi"(%9, %6) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %11 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %12 = "neura.phi"(%11, %5) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %13 = neura.reserve : !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %14 = "neura.phi"(%13, %1) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %15 = neura.reserve : !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %16 = "neura.phi"(%15, %3) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %18 = "neura.phi"(%17, %0) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %19 = neura.reserve : !neura.data<f32, i1>
// FUSE-NEXT:     %20 = "neura.phi"(%19, %8) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// FUSE-NEXT:     %22 = "neura.phi"(%21, %4) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %23 = "neura.gep"(%18, %22) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %24 = "neura.load"(%23) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %25 = "neura.gep"(%16, %22) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %26 = "neura.load"(%25) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     %27 = "neura.fmul_fadd"(%24, %26, %20) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// FUSE-NEXT:     "neura.store"(%27, %14) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// FUSE-NEXT:     %28 = "neura.add"(%22, %12) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// FUSE-NEXT:     %29 = "neura.icmp"(%28, %10) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %30 = "neura.not"(%29) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// FUSE-NEXT:     %31 = neura.grant_predicate %28, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %31 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %32 = neura.grant_predicate %27, %30 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// FUSE-NEXT:     neura.ctrl_mov %32 -> %19 : !neura.data<f32, i1> !neura.data<f32, i1>
// FUSE-NEXT:     %33 = neura.grant_predicate %0, %30 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     neura.ctrl_mov %33 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %34 = neura.grant_predicate %3, %30 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     neura.ctrl_mov %34 -> %15 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %35 = neura.grant_predicate %1, %30 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     neura.ctrl_mov %35 -> %13 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// FUSE-NEXT:     %36 = neura.grant_predicate %5, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %36 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     %37 = neura.grant_predicate %6, %30 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// FUSE-NEXT:     neura.ctrl_mov %37 -> %9 : !neura.data<i64, i1> !neura.data<i64, i1>
// FUSE-NEXT:     "neura.return"() : () -> ()
// FUSE-NEXT:   }

// MOV:        llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// MOV-NEXT:     %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %3 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %4 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %5 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %6 = "neura.grant_once"() <{constant_value = 32 : i64}> : () -> !neura.data<i64, i1>
// MOV-NEXT:     %7 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %8 = "neura.load"(%7) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %9 = "neura.data_mov"(%8) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %10 = "neura.grant_once"(%9) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %11 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %12 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %13 = "neura.phi"(%11, %12) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %15 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %16 = "neura.phi"(%14, %15) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %18 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %19 = "neura.phi"(%17, %18) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %21 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %22 = "neura.phi"(%20, %21) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %23 = neura.reserve : !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %24 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %25 = "neura.phi"(%23, %24) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %26 = neura.reserve : !neura.data<f32, i1>
// MOV-NEXT:     %27 = "neura.data_mov"(%10) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %28 = "neura.phi"(%26, %27) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %29 = neura.reserve : !neura.data<i64, i1>
// MOV-NEXT:     %30 = "neura.data_mov"(%4) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %31 = "neura.phi"(%29, %30) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %32 = "neura.data_mov"(%25) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %33 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %34 = "neura.gep"(%32, %33) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %35 = "neura.data_mov"(%34) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %36 = "neura.load"(%35) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %37 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %38 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %39 = "neura.gep"(%37, %38) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %40 = "neura.data_mov"(%39) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %41 = "neura.load"(%40) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %42 = "neura.data_mov"(%36) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %43 = "neura.data_mov"(%41) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %44 = "neura.data_mov"(%28) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %45 = "neura.fmul_fadd"(%42, %43, %44) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %46 = "neura.data_mov"(%45) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %47 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     "neura.store"(%46, %47) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// MOV-NEXT:     %48 = "neura.data_mov"(%31) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %49 = "neura.data_mov"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %50 = "neura.add"(%48, %49) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %51 = "neura.data_mov"(%50) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %52 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %53 = "neura.icmp"(%51, %52) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %54 = "neura.data_mov"(%53) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %55 = "neura.not"(%54) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %56 = "neura.data_mov"(%50) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %57 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %58 = neura.grant_predicate %56, %57 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %58 -> %29 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %59 = "neura.data_mov"(%45) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MOV-NEXT:     %60 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %61 = neura.grant_predicate %59, %60 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MOV-NEXT:     neura.ctrl_mov %61 -> %26 : !neura.data<f32, i1> !neura.data<f32, i1>
// MOV-NEXT:     %62 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %63 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %64 = neura.grant_predicate %62, %63 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     neura.ctrl_mov %64 -> %23 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %65 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %66 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %67 = neura.grant_predicate %65, %66 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     neura.ctrl_mov %67 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %68 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %69 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %70 = neura.grant_predicate %68, %69 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     neura.ctrl_mov %70 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// MOV-NEXT:     %71 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %72 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %73 = neura.grant_predicate %71, %72 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %73 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     %74 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MOV-NEXT:     %75 = "neura.data_mov"(%55) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MOV-NEXT:     %76 = neura.grant_predicate %74, %75 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MOV-NEXT:     neura.ctrl_mov %76 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
// MOV-NEXT:     "neura.return"() : () -> ()
// MOV-NEXT:   }