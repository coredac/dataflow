// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -O2 -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// Lowers to neura.
// RUN: mlir-neura-opt \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --fuse-patterns \
// RUN:   --insert-mov \
// RUN:   %t-kernel.mlir | FileCheck %s

// Verifies the neura ops are generated. And fusion happens.
// CHECK:      accelerator = "neura"
// CHECK:      "neura.fmul_fadd"
// CHECK:      [[LHS:%.*]] = neura.mov %{{.*}}
// CHECK-NEXT: [[RHS:%.*]] = neura.mov %{{.*}}
// CHECK-NEXT: [[RES:%.*]] = "neura.add"([[LHS]], [[RHS]])
