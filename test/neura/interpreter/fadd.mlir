// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Valid neura.fadd with positive constants
// ===----------------------------------------------------------------------===//
func.func @test_fadd_positive() -> f32 {
  %a = arith.constant 5.5 : f32
  %b = arith.constant 3.25 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fadd:
  // CHECK:   LHS: value = 5.500000e+00, predicate = 1
  // CHECK:   RHS: value = 3.250000e+00, predicate = 1
  // CHECK:   Result: value = 8.750000e+00, predicate = 1
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Valid neura.fadd with negative constants
// ===----------------------------------------------------------------------===//
func.func @test_fadd_negative() -> f32 {
  %a = arith.constant -10.25 : f32
  %b = arith.constant -5.75 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fadd:
  // CHECK:   LHS: value = -1.025000e+01, predicate = 1
  // CHECK:   RHS: value = -5.750000e+00, predicate = 1
  // CHECK:   Result: value = -1.600000e+01, predicate = 1
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Valid neura.fadd with mixed signs
// ===----------------------------------------------------------------------===//
func.func @test_fadd_mixed_signs() -> f32 {
  %a = arith.constant -7.5 : f32
  %b = arith.constant 12.25 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fadd:
  // CHECK:   LHS: value = -7.500000e+00, predicate = 1
  // CHECK:   RHS: value = 1.225000e+01, predicate = 1
  // CHECK:   Result: value = 4.750000e+00, predicate = 1
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Valid neura.fadd with zero
// ===----------------------------------------------------------------------===//
func.func @test_fadd_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 25.5 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: Executing neura.fadd:
  // CHECK:   LHS: value = 0.000000e+00, predicate = 1
  // CHECK:   RHS: value = 2.550000e+01, predicate = 1
  // CHECK:   Result: value = 2.550000e+01, predicate = 1
  return %res : f32
}