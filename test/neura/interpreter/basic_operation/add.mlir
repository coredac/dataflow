// RUN: neura-interpreter %s --verbose | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Add two float constants
// ===----------------------------------------------------------------------===//
func.func @test_add_f32() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 42.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Add a negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_add_negative() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -2.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Add two fractional values
// ===----------------------------------------------------------------------===//
func.func @test_add_fraction() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 1.25 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 4.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Add zero and a number
// ===----------------------------------------------------------------------===//
func.func @test_add_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 7.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 7.000000
  return %res : f32
}

// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 5: Add with operation predicate 0
// ===----------------------------------------------------------------------===//
func.func @test_add_predicate_zero() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %pred = arith.constant 0 : i1
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.add"(%a, %b, %pred_f32) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}