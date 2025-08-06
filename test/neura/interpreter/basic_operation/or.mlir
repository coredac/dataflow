// RUN: neura-interpreter %s --verbose | FileCheck %s

// ====== Bitwise OR Operation Tests ======

// Case 1: 42 | 5 = 47
func.func @test_or_basic() -> i32 {
  %a = arith.constant 42 : i32
  %b = arith.constant 5 : i32
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 47.000000
  return %res : i32
}

// Case 2: OR with zero, should return original number
func.func @test_or_with_zero() -> i32 {
  %a = arith.constant 123 : i32
  %b = arith.constant 0 : i32
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 123.000000
  return %res : i32
}

// Case 3: Self OR, result should equal input
func.func @test_or_self() -> i32 {
  %a = arith.constant 77 : i32
  %res = "neura.or"(%a, %a) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 77.000000
  return %res : i32
}

// Case 4: OR with -1 (all bits set), should return -1
func.func @test_or_with_minus_one() -> i32 {
  %a = arith.constant 123 : i32
  %b = arith.constant -1 : i32
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: -1.000000
  return %res : i32
}