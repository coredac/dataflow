// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura --insert-data-mov %s | FileCheck %s

func.func @test(%a: f32) -> f32 {
  %b = arith.constant 2.0 : f32
  %res = arith.addf %a, %b : f32
  // CHECK: neura.data_mov
  // CHECK: neura.data_mov
  // CHECK: neura.fadd
  return %res : f32
}
