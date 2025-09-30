// RUN: neura-interpreter %s --verbose | FileCheck %s

// Test basic fused fadd operation: (2.5 + 1.5) + 3.0 = 7.0
func.func @test_fadd_fadd_basic() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 1.5 : f32
  %c = arith.constant 3.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 7.000000
  return %res : f32
}

// Test with negative numbers: (5.0 + (-2.0)) + (-1.0) = 2.0
func.func @test_fadd_fadd_negative() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant -2.0 : f32
  %c = arith.constant -1.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 2.000000
  return %res : f32
}

// Test with zero: (0.0 + 4.0) + 6.0 = 10.0
func.func @test_fadd_fadd_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 4.0 : f32
  %c = arith.constant 6.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 10.000000
  return %res : f32
}

// Test with invalid input predicate
func.func @test_fadd_fadd_with_invalid_input_predicate() -> f32 {
  %a = "neura.constant"() {value = 5.0 : f32, predicate = false} : () -> f32
  %b = arith.constant 3.0 : f32
  %c = arith.constant 2.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}