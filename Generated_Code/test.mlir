// RUN: mlir-neura-opt %s | FileCheck %s

func.func @test() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fadd" (%a, %b) : (f32, f32) -> f32

// Checks the expected lowered operation.
// CHECK: neura.fadd

  return %res : f32
}
