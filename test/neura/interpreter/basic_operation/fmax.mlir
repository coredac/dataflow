// RUN: neura-interpreter %s --verbose | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Max of two positive floats
// ===----------------------------------------------------------------------===//
func.func @test_fmax_positive() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = "neura.fmax"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 32.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Max of negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_fmax_mixed() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.fmax"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 3.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Max of two negative floats
// ===----------------------------------------------------------------------===//
func.func @test_fmax_negative() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant -3.0 : f32
  %res = "neura.fmax"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: -3.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Max with zero
// ===----------------------------------------------------------------------===//
func.func @test_fmax_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant -7.0 : f32
  %res = "neura.fmax"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 5: Max of equal values
// ===----------------------------------------------------------------------===//
func.func @test_fmax_equal() -> f32 {
  %a = arith.constant 5.5 : f32
  %b = arith.constant 5.5 : f32
  %res = "neura.fmax"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 5.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 6: Max of fractional values
// ===----------------------------------------------------------------------===//
func.func @test_fmax_fraction() -> f32 {
  %a = arith.constant 2.75 : f32
  %b = arith.constant 2.5 : f32
  %res = "neura.fmax"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter]  → Output: 2.750000
  return %res : f32
}

