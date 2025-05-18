// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir

// Lowers to neura.
// RUN: mlir-neura-opt \
// RUN:   --lower-llvm-to-neura \
// RUN:   --insert-mov %t-kernel.mlir | \
// RUN:   FileCheck %s

// Verifies the neura ops are generated.
// CHECK:      [[LHS:%.*]] = neura.mov %{{.*}}
// CHECK-NEXT: [[RHS:%.*]] = neura.mov %{{.*}}
// CHECK-NEXT: [[RES:%.*]] = "neura.add"([[LHS]], [[RHS]])
