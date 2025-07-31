// Applies pattern fusion before mov insertion.
// RUN: mlir-neura-opt --assign-accelerator --lower-llvm-to-neura --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow --fuse-patterns --insert-data-mov %s | FileCheck %s

func.func @test(%a: f32, %b: f32) -> f32 {
  %c = llvm.mlir.constant(2.0 : f32) : f32
  %temp = llvm.fadd %a, %b : f32
  %res = llvm.fadd %temp, %c : f32
  // CHECK: neura.fadd_fadd
  return %res : f32
}
