// RUN: neura-interpreter %s | FileCheck %s

// Test 1: Bitwise NOT of 42 (result should be -43)
func.func @test_not_basic() -> i32 {
  %a = arith.constant 42 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.not:
  // CHECK-NEXT:   Input: value = 4.200000e+01, predicate = 1
  // CHECK-NEXT:   Bitwise NOT: ~42 = -43
  // CHECK-NEXT:   Final result: value = -4.300000e+01, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: -43.000000
  return %res : i32
}

// Test 2: Bitwise NOT of 0 (result should be -1)
func.func @test_not_zero() -> i32 {
  %a = arith.constant 0 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.not:
  // CHECK-NEXT:   Input: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   Bitwise NOT: ~0 = -1 (0xFFFFFFFFFFFFFFFF)
  // CHECK-NEXT:   Final result: value = -1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: -1.000000
  return %res : i32
}

// Test 3: Bitwise NOT of 1 (result should be -2)
func.func @test_not_one() -> i32 {
  %a = arith.constant 1 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.not:
  // CHECK-NEXT:   Input: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   Bitwise NOT: ~1 = -2
  // CHECK-NEXT:   Final result: value = -2.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: -2.000000
  return %res : i32
}