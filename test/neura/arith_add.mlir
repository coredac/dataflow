// RUN: mlir-neura-opt --lower-arith-to-neura --insert-mov %s | FileCheck %s

func.func @test(%a: f32) -> f32 {
  %b = arith.constant 2.0 : f32
  %res = arith.addf %a, %b : f32
  // CHECK: neura.mov
  // CHECK: neura.mov
  // CHECK: neura.fadd
  return %res : f32
}
