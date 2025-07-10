// RUN: neura-interpreter %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Test 1: Add two float constants
// ===----------------------------------------------------------------------===//
func.func @test_add_f32() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 32.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.add:
  // CHECK-NEXT:   LHS: value = 1.000000e+01, predicate = 1
  // CHECK-NEXT:   RHS: value = 3.200000e+01, predicate = 1
  // CHECK-NEXT:   Result: value = 4.200000e+01, predicate = 1
  // CHECK-NEXT: [neura-interpreter] Output: 42.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 2: Add a negative and positive float
// ===----------------------------------------------------------------------===//
func.func @test_add_negative() -> f32 {
  %a = arith.constant -5.0 : f32
  %b = arith.constant 3.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.add:
  // CHECK-NEXT:   LHS: value = -5.000000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Result: value = -2.000000e+00, predicate = 1
  // CHECK-NEXT: [neura-interpreter] Output: -2.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 3: Add two fractional values
// ===----------------------------------------------------------------------===//
func.func @test_add_fraction() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 1.25 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.add:
  // CHECK-NEXT:   LHS: value = 2.500000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 1.250000e+00, predicate = 1
  // CHECK-NEXT:   Result: value = 4.000000e+00, predicate = 1
  // CHECK-NEXT: [neura-interpreter] Output: 4.000000
  return %res : f32
}

// ===----------------------------------------------------------------------===//
// Test 4: Add zero and a number
// ===----------------------------------------------------------------------===//
func.func @test_add_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 7.0 : f32
  %res = "neura.add"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.add:
  // CHECK-NEXT:   LHS: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   RHS: value = 7.000000e+00, predicate = 1
  // CHECK-NEXT:   Result: value = 7.000000e+00, predicate = 1
  // CHECK-NEXT: [neura-interpreter] Output: 7.000000
  return %res : f32
}