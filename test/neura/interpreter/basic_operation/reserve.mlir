// RUN: neura-interpreter %s | FileCheck %s

func.func @test_reserve_basic() {
  %a = "neura.reserve"() : () -> (i32)
  // CHECK: [neura-interpreter]  → Output: (void)
  return
}

func.func @test_reserve_multiple_types() {
  %a = "neura.reserve"() : () -> (i32)
  %b = "neura.reserve"() : () -> (f32)
  %c = "neura.reserve"() : () -> (tensor<4xf32>)
  // CHECK: [neura-interpreter]  → Output: (void)
  return
}