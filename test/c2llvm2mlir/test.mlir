// Compiles the original kernel.
// RUN: clang++ kernel.cpp -o %t-kernel.out

// Compiles the original kernel to mlir, then lower back to llvm, eventually binary.
// RUN: clang++ -S -emit-llvm -o %t-kernel.ll kernel.cpp
// RUN: mlir-translate --import-llvm %t-kernel.ll -o %t-kernel.mlir
// RUN: mlir-opt %t-kernel.mlir | mlir-translate -mlir-to-llvmir -o %t-kernel_back.ll
// RUN: llc %t-kernel_back.ll -filetype=obj -o %t-kernel_back.o
// RUN: clang++ %t-kernel_back.o -o %t-kernel_back.out

// RUN: %t-kernel.out > %t-dumped_output.txt
// RUN: %t-kernel_back.out >> %t-dumped_output.txt
// RUN: FileCheck %s < %t-dumped_output.txt

// Verifies the output values are the same for the original and re-compiled kernel.
// CHECK: output: [[OUTPUT:[0-9]+\.[0-9]+]]
// CHECK: output: [[OUTPUT]]
