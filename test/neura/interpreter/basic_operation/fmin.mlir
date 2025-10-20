// RUN: neura-interpreter %s --verbose | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Min of two positive floats
// ===----------------------------------------------------------------------===//
func.func @test_fmin_positive() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = "neura.fmin"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 10.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Min of negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_fmin_mixed() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.fmin"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -5.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Min of two negative floats
// ===----------------------------------------------------------------------===//
func.func @test_fmin_negative() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant -3.0 : f32
  %res = "neura.fmin"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -10.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Min with zero
// ===----------------------------------------------------------------------===//
func.func @test_fmin_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 7.0 : f32
  %res = "neura.fmin"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 5: Min of equal values
// ===----------------------------------------------------------------------===//
func.func @test_fmin_equal() -> f32 {
  %a = arith.constant 5.5 : f32
  %b = arith.constant 5.5 : f32
  %res = "neura.fmin"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 5.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 6: Min of fractional values
// ===----------------------------------------------------------------------===//
func.func @test_fmin_fraction() -> f32 {
  %a = arith.constant 2.75 : f32
  %b = arith.constant 2.5 : f32
  %res = "neura.fmin"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 2.500000
  return %res : f32
}

