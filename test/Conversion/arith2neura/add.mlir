// RUN: neura-compiler --neura-conversion %s | FileCheck %s --check-prefix=COMPILER
// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura %s | FileCheck %s --check-prefix=OPT

func.func @test(%a: f32) -> f32 {
  %b = arith.constant 2.0 : f32
  %res = arith.addf %a, %b : f32
  return %res : f32
}

// COMPILER: neura.fadd
// OPT:      neura.fadd