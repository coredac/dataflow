// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Valid neura.fadd with positive constants
// ===----------------------------------------------------------------------===//
func.func @test_fadd_positive() -> f32 {
  %a = arith.constant 5.5 : f32
  %b = arith.constant 3.25 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 8.750000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Valid neura.fadd with negative constants
// ===----------------------------------------------------------------------===//
func.func @test_fadd_negative() -> f32 {
  %a = arith.constant -10.25 : f32
  %b = arith.constant -5.75 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -16.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Valid neura.fadd with mixed signs
// ===----------------------------------------------------------------------===//
func.func @test_fadd_mixed_signs() -> f32 {
  %a = arith.constant -7.5 : f32
  %b = arith.constant 12.25 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 4.750000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Valid neura.fadd with zero
// ===----------------------------------------------------------------------===//
func.func @test_fadd_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 25.5 : f32
  %res = "neura.fadd"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 25.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Predicate handling in neura.fadd
// ===----------------------------------------------------------------------===//
func.func @test_fadd_invalid_predicate() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 25.5 : f32
  %pred = arith.constant 0 : i1
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fadd"(%a, %b, %pred_f32) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 5: Nested predicate handling in neura.fadd
// ===----------------------------------------------------------------------===//
func.func @test_nested_fadd_invalid_predicate() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 25.5 : f32
  %pred = arith.constant 0 : i1
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %tmp = "neura.fadd"(%a, %b, %pred_f32) : (f32, f32, f32) -> f32
  %res = "neura.fadd"(%tmp, %b, %pred_f32) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}