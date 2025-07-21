// RUN: neura-interpreter %s | FileCheck %s

// Test 1: Bitwise NOT of 42 (result should be -43)
func.func @test_not_basic() -> i32 {
  %a = arith.constant 42 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: -43.000000
  return %res : i32
}

// Test 2: Bitwise NOT of 0 (result should be -1)
func.func @test_not_zero() -> i32 {
  %a = arith.constant 0 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: -1.000000
  return %res : i32
}

// Test 3: Bitwise NOT of 1 (result should be -2)
func.func @test_not_one() -> i32 {
  %a = arith.constant 1 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: -2.000000
  return %res : i32
}