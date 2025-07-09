// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Add two float constants
// ===----------------------------------------------------------------------===//
func.func @test_add_f32() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.add:
  // CHECK:   LHS: value = 10
  // CHECK:   RHS: value = 32
  // CHECK:   Result: value = 42
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Add a negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_add_negative() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.add:
  // CHECK:   LHS: value = -5
  // CHECK:   RHS: value = 3
  // CHECK:   Result: value = -2
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Add two fractional values
// ===----------------------------------------------------------------------===//
func.func @test_add_fraction() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 1.25 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.add:
  // CHECK:   LHS: value = 2.5
  // CHECK:   RHS: value = 1.25
  // CHECK:   Result: value = 3.75
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Add zero and a number
// ===----------------------------------------------------------------------===//
func.func @test_add_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 7.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.add:
  // CHECK:   LHS: value = 0
  // CHECK:   RHS: value = 7
  // CHECK:   Result: value = 7
  return %res : f32
}
