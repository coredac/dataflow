// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O1 -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// TODO: Enable --insert-mov once the backward ctrl flow mov is supported.
// Lowers to neura.
// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:  | FileCheck %s

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fuse-patterns \
// RUN:  | FileCheck %s --check-prefix=CHECK-FUSED

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --neura-canonicalize \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fuse-patterns \
// RUN:   --insert-data-mov \
// RUN:  | FileCheck %s --check-prefix=CHECK-MOV

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
// CHECK-FUSED: llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-FUSED:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %1 = "neura.grant_always"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %2 = "neura.grant_once"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %3 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %4 = "neura.grant_always"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %5 = "neura.grant_once"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %6 = "neura.constant"() <{predicate = true, value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %8 = "neura.grant_once"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %10 = "neura.grant_once"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %11 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %12 = "neura.grant_always"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %13 = "neura.grant_once"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %14 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %15 = "neura.grant_always"(%14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %16 = "neura.grant_once"(%14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %17 = "neura.load"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %18 = "neura.grant_once"(%17) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %19 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %20 = "neura.phi"(%19, %16) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %21 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %22 = "neura.phi"(%21, %13) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %23 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %24 = "neura.phi"(%23, %5) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %25 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %26 = "neura.phi"(%25, %8) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %27 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %28 = "neura.phi"(%27, %2) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %29 = neura.reserve : !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %30 = "neura.phi"(%29, %18) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %31 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %32 = "neura.phi"(%31, %10) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %33 = "neura.gep"(%28, %32) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %34 = "neura.load"(%33) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %35 = "neura.gep"(%26, %32) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %36 = "neura.load"(%35) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %37 = "neura.fmul_fadd"(%34, %36, %30) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     "neura.store"(%37, %24) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-FUSED-NEXT:     %38 = "neura.add"(%32, %22) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %39 = "neura.icmp"(%38, %20) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     %40 = "neura.not"(%39) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     %41 = neura.grant_predicate %38, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %41 -> %31 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %42 = neura.grant_predicate %37, %40 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %42 -> %29 : !neura.data<f32, i1> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %43 = neura.grant_predicate %2, %40 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %43 -> %27 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %44 = neura.grant_predicate %8, %40 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %44 -> %25 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %45 = neura.grant_predicate %5, %40 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %45 -> %23 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %46 = neura.grant_predicate %13, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %46 -> %21 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %47 = neura.grant_predicate %16, %40 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %47 -> %19 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     "neura.return"() : () -> ()
// CHECK-FUSED-NEXT:   }

// CHECK-MOV: llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-MOV-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %1 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %2 = "neura.grant_always"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %3 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %4 = "neura.grant_once"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %5 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %6 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %7 = "neura.grant_always"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %8 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %10 = "neura.constant"() <{predicate = true, value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %11 = "neura.data_mov"(%10) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %12 = "neura.grant_always"(%11) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %13 = "neura.data_mov"(%10) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %14 = "neura.grant_once"(%13) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %15 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %16 = "neura.data_mov"(%15) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %18 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %19 = "neura.data_mov"(%18) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %20 = "neura.grant_always"(%19) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %21 = "neura.data_mov"(%18) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %22 = "neura.grant_once"(%21) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %23 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %24 = "neura.data_mov"(%23) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %25 = "neura.grant_always"(%24) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %26 = "neura.data_mov"(%23) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %27 = "neura.grant_once"(%26) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %28 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %29 = "neura.load"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %30 = "neura.data_mov"(%29) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %31 = "neura.grant_once"(%30) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %32 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %33 = "neura.data_mov"(%27) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %34 = "neura.phi"(%32, %33) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %35 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %36 = "neura.data_mov"(%22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %37 = "neura.phi"(%35, %36) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %38 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %39 = "neura.data_mov"(%9) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %40 = "neura.phi"(%38, %39) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %41 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %42 = "neura.data_mov"(%14) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %43 = "neura.phi"(%41, %42) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %44 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %45 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %46 = "neura.phi"(%44, %45) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %47 = neura.reserve : !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %48 = "neura.data_mov"(%31) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %49 = "neura.phi"(%47, %48) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %50 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %51 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %52 = "neura.phi"(%50, %51) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %53 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %54 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %55 = "neura.gep"(%53, %54) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %56 = "neura.data_mov"(%55) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %57 = "neura.load"(%56) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %58 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %59 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %60 = "neura.gep"(%58, %59) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %61 = "neura.data_mov"(%60) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %62 = "neura.load"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %63 = "neura.data_mov"(%57) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %64 = "neura.data_mov"(%62) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %65 = "neura.data_mov"(%49) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %66 = "neura.fmul_fadd"(%63, %64, %65) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %67 = "neura.data_mov"(%66) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %68 = "neura.data_mov"(%40) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     "neura.store"(%67, %68) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-MOV-NEXT:     %69 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %70 = "neura.data_mov"(%37) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %71 = "neura.add"(%69, %70) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %72 = "neura.data_mov"(%71) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %73 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %74 = "neura.icmp"(%72, %73) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %75 = "neura.data_mov"(%74) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %76 = "neura.not"(%75) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %77 = "neura.data_mov"(%71) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %78 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %79 = neura.grant_predicate %77, %78 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %79 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %80 = "neura.data_mov"(%66) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %81 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %82 = neura.grant_predicate %80, %81 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %82 -> %47 : !neura.data<f32, i1> !neura.data<f32, i1>
// CHECK-MOV-NEXT:     %83 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %84 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %85 = neura.grant_predicate %83, %84 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %85 -> %44 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %86 = "neura.data_mov"(%14) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %87 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %88 = neura.grant_predicate %86, %87 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %88 -> %41 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %89 = "neura.data_mov"(%9) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %90 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %91 = neura.grant_predicate %89, %90 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %91 -> %38 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:     %92 = "neura.data_mov"(%22) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %93 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %94 = neura.grant_predicate %92, %93 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %94 -> %35 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %95 = "neura.data_mov"(%27) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     %96 = "neura.data_mov"(%76) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:     %97 = neura.grant_predicate %95, %96 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     neura.ctrl_mov %97 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:     "neura.return"() : () -> ()
// CHECK-MOV-NEXT:   }