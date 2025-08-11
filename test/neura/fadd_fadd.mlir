// Applies pattern fusion before mov insertion.
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-pattern --insert-data-mov %s | FileCheck %s

func.func @test(%a: f32, %b: f32) -> f32 {
  %c = arith.constant 2.0 : f32
  %temp = arith.addf %a, %b : f32
  %res = arith.addf %temp, %c : f32
  // CHECK: neura.fadd_fadd
  return %res : f32
}