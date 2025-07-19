// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O1 -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// TODO: Enable --insert-mov once the backward ctrl flow mov is supported.
// Lowers to neura.
// TODO: Make `--leverage-predicated-value` work. Segmentation fault for now.
// https://github.com/coredac/dataflow/issues/84.
// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fuse-patterns \
// RN:   --insert-mov \
// RUN:   %t-kernel.mlir > %t-lowered.mlir

// RUN: FileCheck %s < %t-lowered.mlir

// Verifies the neura ops are generated. And fusion happens.
// CHECK:      accelerator = "neura"
// CHECK-NEXT: %0 = "neura.constant"() <{predicate = true, value = 0 : i64}> : () -> i64
// CHECK-NEXT: %1 = "neura.constant"() <{predicate = true, value = 1 : i64}> : () -> i64
// CHECK-NEXT: %2 = "neura.constant"() <{predicate = true, value = 32 : i64}> : () -> i64
// CHECK-NEXT: %3 = neura.reserve : i64
// CHECK-NEXT: %4 = "neura.phi"(%3, %0) : (i64, i64) -> i64
// CHECK-NEXT: %5 = "neura.gep"(%arg0, %4) : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT: %6 = "neura.load"(%5) : (!llvm.ptr) -> f32
// CHECK-NEXT: %7 = "neura.gep"(%arg2, %4) : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-NEXT: %8 = "neura.load"(%7) : (!llvm.ptr) -> f32
// CHECK-NEXT: %9 = "neura.load"(%arg1) : (!llvm.ptr) -> f32
// CHECK-NEXT: %10 = "neura.fmul_fadd"(%6, %8, %9) : (f32, f32, f32) -> f32
// CHECK-NEXT: "neura.store"(%10, %arg1) : (f32, !llvm.ptr) -> ()
// CHECK-NEXT: %11 = "neura.add"(%4, %1) : (i64, i64) -> i64
// CHECK-NEXT: %12 = "neura.icmp"(%11, %2) <{cmpType = "eq"}> : (i64, i64) -> i1
// CHECK-NEXT: %13 = "neura.not"(%12) : (i1) -> i1
// CHECK-NEXT: %14 = neura.grant_predicate %11, %13 : i64, i1 -> i64
// CHECK-NEXT: neura.ctrl_mov %14 -> %3 : i64 i64
// CHECK-NEXT: "neura.return"() : () -> ()