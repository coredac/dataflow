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
// RUN:   --fuse-patterns \
// RUN:  | FileCheck %s --check-prefix=CHECK-FUSED

// RUN: mlir-neura-opt %t-kernel.mlir\
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
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
// CHECK-FUSED:     llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-FUSED-NEXT:     %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %1 = "neura.grant_once"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %2 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %3 = "neura.grant_once"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %4 = "neura.constant"() <{predicate = true, value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %5 = "neura.grant_once"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %7 = "neura.grant_once"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %8 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %9 = "neura.grant_once"(%8) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %10 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %12 = "neura.load"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %13 = "neura.grant_once"(%12) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %14 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %15 = "neura.phi"(%14, %11) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %16 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %17 = "neura.phi"(%16, %9) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %18 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %19 = "neura.phi"(%18, %3) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %21 = "neura.phi"(%20, %5) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %22 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %23 = "neura.phi"(%22, %1) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %24 = neura.reserve : !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %25 = "neura.phi"(%24, %13) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %26 = neura.reserve : !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %27 = "neura.phi"(%26, %7) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %28 = "neura.gep"(%23, %27) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %29 = "neura.load"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %30 = "neura.gep"(%21, %27) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %31 = "neura.load"(%30) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %32 = "neura.fmul_fadd"(%29, %31, %25) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     "neura.store"(%32, %19) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-FUSED-NEXT:     %33 = "neura.add"(%27, %17) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %34 = "neura.icmp"(%33, %15) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     %35 = "neura.not"(%34) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-FUSED-NEXT:     %36 = neura.grant_predicate %33, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %36 -> %26 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %37 = neura.grant_predicate %32, %35 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %37 -> %24 : !neura.data<f32, i1> !neura.data<f32, i1>
// CHECK-FUSED-NEXT:     %38 = neura.grant_predicate %1, %35 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %38 -> %22 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %39 = neura.grant_predicate %5, %35 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %39 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %40 = neura.grant_predicate %3, %35 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %40 -> %18 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-FUSED-NEXT:     %41 = neura.grant_predicate %9, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %41 -> %16 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     %42 = neura.grant_predicate %11, %35 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     neura.ctrl_mov %42 -> %14 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-FUSED-NEXT:     "neura.return"() : () -> ()
// CHECK-FUSED-NEXT:   }

// CHECK-MOV:     llvm.func local_unnamed_addr @_Z6kernelPfS_S_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) attributes {accelerator = "neura", memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
// CHECK-MOV-NEXT:         %0 = "neura.constant"() <{predicate = true, value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %1 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %2 = "neura.grant_once"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %3 = "neura.constant"() <{predicate = true, value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %4 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %5 = "neura.grant_once"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %6 = "neura.constant"() <{predicate = true, value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %7 = "neura.data_mov"(%6) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %8 = "neura.grant_once"(%7) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %9 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %10 = "neura.data_mov"(%9) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %11 = "neura.grant_once"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %12 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %13 = "neura.data_mov"(%12) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %14 = "neura.grant_once"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %15 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %16 = "neura.data_mov"(%15) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %17 = "neura.grant_once"(%16) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %18 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %19 = "neura.load"(%18) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %20 = "neura.data_mov"(%19) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %21 = "neura.grant_once"(%20) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %22 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %23 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %24 = "neura.phi"(%22, %23) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %25 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %26 = "neura.data_mov"(%14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %27 = "neura.phi"(%25, %26) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %28 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %29 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %30 = "neura.phi"(%28, %29) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %31 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %32 = "neura.data_mov"(%8) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %33 = "neura.phi"(%31, %32) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %34 = neura.reserve : !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %35 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %36 = "neura.phi"(%34, %35) : (!neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %37 = neura.reserve : !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %38 = "neura.data_mov"(%21) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %39 = "neura.phi"(%37, %38) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %40 = neura.reserve : !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %41 = "neura.data_mov"(%11) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %42 = "neura.phi"(%40, %41) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %43 = "neura.data_mov"(%36) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %44 = "neura.data_mov"(%42) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %45 = "neura.gep"(%43, %44) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %46 = "neura.data_mov"(%45) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %47 = "neura.load"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %48 = "neura.data_mov"(%33) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %49 = "neura.data_mov"(%42) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %50 = "neura.gep"(%48, %49) : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %51 = "neura.data_mov"(%50) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %52 = "neura.load"(%51) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %53 = "neura.data_mov"(%47) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %54 = "neura.data_mov"(%52) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %55 = "neura.data_mov"(%39) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %56 = "neura.fmul_fadd"(%53, %54, %55) : (!neura.data<f32, i1>, !neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %57 = "neura.data_mov"(%56) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %58 = "neura.data_mov"(%30) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         "neura.store"(%57, %58) : (!neura.data<f32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
// CHECK-MOV-NEXT:         %59 = "neura.data_mov"(%42) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %60 = "neura.data_mov"(%27) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %61 = "neura.add"(%59, %60) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %62 = "neura.data_mov"(%61) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %63 = "neura.data_mov"(%24) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %64 = "neura.icmp"(%62, %63) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %65 = "neura.data_mov"(%64) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %66 = "neura.not"(%65) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %67 = "neura.data_mov"(%61) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %68 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %69 = neura.grant_predicate %67, %68 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %69 -> %40 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %70 = "neura.data_mov"(%56) : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %71 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %72 = neura.grant_predicate %70, %71 : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %72 -> %37 : !neura.data<f32, i1> !neura.data<f32, i1>
// CHECK-MOV-NEXT:         %73 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %74 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %75 = neura.grant_predicate %73, %74 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %75 -> %34 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %76 = "neura.data_mov"(%8) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %77 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %78 = neura.grant_predicate %76, %77 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %78 -> %31 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %79 = "neura.data_mov"(%5) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %80 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %81 = neura.grant_predicate %79, %80 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %81 -> %28 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
// CHECK-MOV-NEXT:         %82 = "neura.data_mov"(%14) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %83 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %84 = neura.grant_predicate %82, %83 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %84 -> %25 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %85 = "neura.data_mov"(%17) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         %86 = "neura.data_mov"(%66) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// CHECK-MOV-NEXT:         %87 = neura.grant_predicate %85, %86 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         neura.ctrl_mov %87 -> %22 : !neura.data<i64, i1> !neura.data<i64, i1>
// CHECK-MOV-NEXT:         "neura.return"() : () -> ()
// CHECK-MOV-NEXT:       }