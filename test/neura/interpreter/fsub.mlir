// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Valid neura.fsub with positive constants
// ===----------------------------------------------------------------------===//
func.func @test_fsub_positive() -> f32 {
  %a = arith.constant 10.5 : f32
  %b = arith.constant 3.25 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fsub:
  // CHECK:   LHS: value = 10.5
  // CHECK:   RHS: value = 3.25
  // CHECK:   Result: value = 7.25
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Valid neura.fsub with negative result
// ===----------------------------------------------------------------------===//
func.func @test_fsub_negative_result() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 8.75 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fsub:
  // CHECK:   LHS: value = 5
  // CHECK:   RHS: value = 8.75
  // CHECK:   Result: value = -3.75
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Valid neura.fsub with negative operands
// ===----------------------------------------------------------------------===//
func.func @test_fsub_negative_operands() -> f32 {
  %a = arith.constant -5.25 : f32
  %b = arith.constant -3.75 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fsub:
  // CHECK:   LHS: value = -5.25
  // CHECK:   RHS: value = -3.75
  // CHECK:   Result: value = -1.5
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Valid neura.fsub with zero
// ===----------------------------------------------------------------------===//
func.func @test_fsub_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 10.5 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fsub:
  // CHECK:   LHS: value = 0
  // CHECK:   RHS: value = 10.5
  // CHECK:   Result: value = -10.5
  return %res : f32
}
