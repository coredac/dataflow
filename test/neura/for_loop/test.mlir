// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O2 -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// TODO: Enable --insert-mov once the backward ctrl flow mov is supported.
// Lowers to neura.
// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fuse-patterns \
// RN:   --insert-mov \
// RUN:   %t-kernel.mlir | FileCheck %s

// Verifies the neura ops are generated. And fusion happens.
// CHECK: accelerator = "neura"
// CHECK-NOT: = llvm.
