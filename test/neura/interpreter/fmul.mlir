// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Valid neura.fmul with positive constants
// ===----------------------------------------------------------------------===//
func.func @test_fmul_positive() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul:
  // CHECK: LHS: value = 2.500000e+00, predicate = 1
  // CHECK: RHS: value = 3.000000e+00, predicate = 1
  // CHECK: Result: value = 7.500000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 7.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Valid neura.fmul with negative result
// ===----------------------------------------------------------------------===//
func.func @test_fmul_negative_result() -> f32 {
  %a = arith.constant -2.5 : f32
  %b = arith.constant 4.0 : f32
  %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul:
  // CHECK: LHS: value = -2.500000e+00, predicate = 1
  // CHECK: RHS: value = 4.000000e+00, predicate = 1
  // CHECK: Result: value = -1.000000e+01, predicate = 1
  // CHECK: [neura-interpreter] Output: -10.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Valid neura.fmul with zero
// ===----------------------------------------------------------------------===//
func.func @test_fmul_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 10.5 : f32
  %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul:
  // CHECK: LHS: value = 0.000000e+00, predicate = 1
  // CHECK: RHS: value = 1.050000e+01, predicate = 1
  // CHECK: Result: value = 0.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Valid neura.fmul with three operands (including predicate)
// ===----------------------------------------------------------------------===//
func.func @test_fmul_with_predicate() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 5.0 : f32
  %p = arith.constant 1.0 : f32  // Non-zero predicate
  %res = "neura.fmul"(%a, %b, %p) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul:
  // CHECK: LHS: value = 2.000000e+00, predicate = 1
  // CHECK: RHS: value = 5.000000e+00, predicate = 1
  // CHECK: Predicate: value = 1.000000e+00, predicate = 1
  // CHECK: Result: value = 1.000000e+01, predicate = 1
  // CHECK: [neura-interpreter] Output: 10.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 5: Neura.fmul with zero predicate (Negative case)
// ===----------------------------------------------------------------------===//
func.func @test_fmul_zero_predicate() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 5.0 : f32
  %p = arith.constant 0.0 : f32  // Zero predicate
  %res = "neura.fmul"(%a, %b, %p) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul:
  // CHECK: LHS: value = 2.000000e+00, predicate = 1
  // CHECK: RHS: value = 5.000000e+00, predicate = 1
  // CHECK: Predicate: value = 0.000000e+00, predicate = 1
  // CHECK: Result: value = 1.000000e+01, predicate = 0
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}