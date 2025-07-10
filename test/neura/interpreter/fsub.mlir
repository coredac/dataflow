// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Valid neura.fsub with positive constants
// ===----------------------------------------------------------------------===//
func.func @test_fsub_positive() -> f32 {
  %a = arith.constant 10.5 : f32
  %b = arith.constant 3.25 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fsub:
  // CHECK:  LHS: value = 1.050000e+01, predicate = 1
  // CHECK:  RHS: value = 3.250000e+00, predicate = 1
  // CHECK:  Result: value = 7.250000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 7.250000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Valid neura.fsub with negative result
// ===----------------------------------------------------------------------===//
func.func @test_fsub_negative_result() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 8.75 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fsub:
  // CHECK:  LHS: value = 5.000000e+00, predicate = 1
  // CHECK:  RHS: value = 8.750000e+00, predicate = 1
  // CHECK:  Result: value = -3.750000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: -3.750000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Valid neura.fsub with negative operands
// ===----------------------------------------------------------------------===//
func.func @test_fsub_negative_operands() -> f32 {
  %a = arith.constant -5.25 : f32
  %b = arith.constant -3.75 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fsub:
  // CHECK:  LHS: value = -5.250000e+00, predicate = 1
  // CHECK:  RHS: value = -3.750000e+00, predicate = 1
  // CHECK:  Result: value = -1.500000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: -1.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Valid neura.fsub with zero
// ===----------------------------------------------------------------------===//
func.func @test_fsub_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 10.5 : f32
  %res = "neura.fsub"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fsub:
  // CHECK:  LHS: value = 0.000000e+00, predicate = 1
  // CHECK:  RHS: value = 1.050000e+01, predicate = 1
  // CHECK:  Result: value = -1.050000e+01, predicate = 1
  // CHECK: [neura-interpreter] Output: -10.500000
  return %res : f32
}