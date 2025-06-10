// RUN: neura-interpreter %s | FileCheck %s

module {
  func.func @test() -> f32 {
    %arg0 = "neura.constant"() <{value = 9.0 : f32}> : () -> f32
    %cst = "neura.constant"() <{value = 2.0 : f32}> : () -> f32
    %0 = "neura.data_mov"(%arg0) : (f32) -> f32
    %1 = "neura.data_mov"(%cst) : (f32) -> f32
    %2 = "neura.fadd"(%0, %1) : (f32, f32) -> f32
    return %2 : f32
    // CHECK: 11.0
  }
}
