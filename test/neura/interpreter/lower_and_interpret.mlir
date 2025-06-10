// RUN: mlir-opt %s \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-math-to-llvm \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN:   -o %t-lowered-to-llvm.mlir

// RUN: mlir-translate -mlir-to-llvmir \
// RUN:   %t-lowered-to-llvm.mlir \
// RUN:   -o %t-lower_and_interpreter.ll

// RUN: llc %t-lower_and_interpreter.ll \
// RUN:   -filetype=obj -o %t-out.o

// RUN: clang++ main.cpp %t-out.o \
// RUN:   -o %t-out.bin

// RUN: %t-out.bin > %t-dumped_output.txt

// RUN: mlir-neura-opt --lower-arith-to-neura --insert-data-mov %s \
// RUN:   -o %t-neura.mlir

// RUN: neura-interpreter %t-neura.mlir >> %t-dumped_output.txt

// RUN: FileCheck %s < %t-dumped_output.txt

module {
  func.func @test() -> f32 attributes { llvm.emit_c_interface }{
    %arg0 = arith.constant 9.0 : f32
    %cst = arith.constant 2.0 : f32
    %0 = arith.addf %arg0, %cst : f32
    // CHECK: Golden output: [[OUTPUT:[0-9]+\.[0-9]+]]
    // CHECK: [neura-interpreter] Output: [[OUTPUT]]
    return %0 : f32
  }
}

