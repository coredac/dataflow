// RUN: mlir-neura-opt %s | FileCheck %s

func.func @test_reserve_basic() {
  %a = "neura.reserve"() : () -> (i32)
  // CHECK: neura.reserve: created placeholder
  // CHECK: Type: i32
  
  return
}

func.func @test_reserve_multiple_types() {
  %a = "neura.reserve"() : () -> (i32)
  %b = "neura.reserve"() : () -> (f32)
  %c = "neura.reserve"() : () -> (tensor<4xf32>)
  // CHECK: neura.reserve: created placeholder
  // CHECK-NEXT: Type: i32
  // CHECK-NEXT: neura.reserve: created placeholder
  // CHECK-NEXT: Type: f32
  // CHECK-NEXT: neura.reserve: created placeholder
  // CHECK-NEXT: Type: tensor<4xf32>
  
  return
}
