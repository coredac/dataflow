// RUN: neura-interpreter %s --verbose | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Valid neura.fmul with positive constants
// ===----------------------------------------------------------------------===//
func.func @test_fmul_positive() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 7.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Valid neura.fmul with negative result
// ===----------------------------------------------------------------------===//
func.func @test_fmul_negative_result() -> f32 {
  %a = arith.constant -2.5 : f32
  %b = arith.constant 4.0 : f32
  %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -10.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Valid neura.fmul with zero
// ===----------------------------------------------------------------------===//
func.func @test_fmul_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 10.5 : f32
  %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Valid neura.fmul with three operands (including predicate)
// ===----------------------------------------------------------------------===//
// func.func @test_fmul_with_embed_predicate() -> f32 {
//   %a = "neura.constant"() {value = 2.0 : f32, predicate = true} : () -> f32
//   %b = "neura.constant"() {value = 5.0 : f32, predicate = true} : () -> f32
//   %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
//   // [neura-interpreter]  → Output: 10.000000
//   return %res : f32
// }

// ===----------------------------------------------------------------------===//
// Test 5: Neura.fmul with zero predicate (Negative case)
// ===----------------------------------------------------------------------===//
// func.func @test_fmul_embed_zero_predicate() -> f32 {
//   %a = "neura.constant"() {value = 2.0 : f32, predicate = false} : () -> f32
//   %b = "neura.constant"() {value = 5.0 : f32, predicate = false} : () -> f32
//   %res = "neura.fmul"(%a, %b) : (f32, f32) -> f32
//   // [neura-interpreter]  → Output: 0.000000
//   return %res : f32
// }