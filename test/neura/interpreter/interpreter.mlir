// RUN: neura-interpreter %s | FileCheck %s

module {
  func.func @test() -> f32 {
    %arg0 = arith.constant 9.0 : f32
    %cst = arith.constant 2.0 : f32
    %0 = neura.mov %arg0 : f32 -> f32
    %1 = neura.mov %cst : f32 -> f32
    %2 = "neura.fadd"(%0, %1) : (f32, f32) -> f32
    return %2 : f32
    // CHECK: 1.1
  }
}
