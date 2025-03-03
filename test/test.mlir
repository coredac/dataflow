// RUN: mlir-neura-opt %s | FileCheck %s

func.func @test() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %res = neura.add %a, %b : f32

// Checks the expected lowered operation.
// CHECK: neura.add

  return %res : f32
}
