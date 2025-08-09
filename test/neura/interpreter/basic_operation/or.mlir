// RUN: neura-interpreter %s --verbose | FileCheck %s

// ====== Logical OR Operation Tests ======

// Case 1: true | true = true (1 || 1 = 1)
func.func @test_or_basic_true_true() -> i32 {
%a = arith.constant 1 : i32 // true (non-zero)
  %b = arith.constant 1 : i32 // true (non-zero)
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : i32
}

// Case 2: true | false = true (1 || 0 = 1)
func.func @test_or_true_false() -> i32 {
  %a = arith.constant 1 : i32 // true
  %b = arith.constant 0 : i32 // false
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : i32
}

// Case 3: false | true = true (0 || 5 = 1)
func.func @test_or_false_true() -> i32 {
  %a = arith.constant 0 : i32 // false
  %b = arith.constant 5 : i32 // true (non-zero)
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : i32
}

// Case 4: false | false = false (0 || 0 = 0)
func.func @test_or_false_false() -> i32 {
  %a = arith.constant 0 : i32 // false
  %b = arith.constant 0 : i32 // false
  %res = "neura.or"(%a, %b) : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : i32
}