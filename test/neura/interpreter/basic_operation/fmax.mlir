// RUN: neura-interpreter %s --verbose | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Max of two positive floats
// ===----------------------------------------------------------------------===//
func.func @test_fmax_positive() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 32.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Max of negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_fmax_mixed() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 3.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Max of two negative floats
// ===----------------------------------------------------------------------===//
func.func @test_fmax_negative() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant -3.0 : f32
  %res = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: -3.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Max with zero
// ===----------------------------------------------------------------------===//
func.func @test_fmax_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant -7.0 : f32
  %res = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 5: Max of equal values
// ===----------------------------------------------------------------------===//
func.func @test_fmax_equal() -> f32 {
  %a = arith.constant 5.5 : f32
  %b = arith.constant 5.5 : f32
  %res = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 5.500000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 6: Max of fractional values
// ===----------------------------------------------------------------------===//
func.func @test_fmax_fraction() -> f32 {
  %a = arith.constant 2.75 : f32
  %b = arith.constant 2.5 : f32
  %res = neura.fmax<"maxnum">(%a, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 2.750000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 7: FMax with NaN (maxnum semantic)
// ===----------------------------------------------------------------------===//
func.func @test_fmax_nan_maxnum_lhs() -> f32 {
  %nan = arith.constant 0x7FC00000 : f32  // NaN
  %b = arith.constant 5.0 : f32
  %res = neura.fmax<"maxnum">(%nan, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 5.000000
  return %res : f32
}

func.func @test_fmax_nan_maxnum_rhs() -> f32 {
  %a = arith.constant 5.0 : f32
  %nan = arith.constant 0x7FC00000 : f32  // NaN
  %res = neura.fmax<"maxnum">(%a, %nan : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: 5.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 8: FMax with NaN (maximum semantic)
// ===----------------------------------------------------------------------===//
func.func @test_fmax_nan_maximum_lhs() -> f32 {
  %nan = arith.constant 0x7FC00000 : f32  // NaN
  %b = arith.constant 5.0 : f32
  %res = neura.fmax<"maximum">(%nan, %b : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: nan
  return %res : f32
}

func.func @test_fmax_nan_maximum_rhs() -> f32 {
  %a = arith.constant 5.0 : f32
  %nan = arith.constant 0x7FC00000 : f32  // NaN
  %res = neura.fmax<"maximum">(%a, %nan : f32) : f32 -> f32
  // CHECK: [neura-interpreter]  → Output: nan
  return %res : f32
}

