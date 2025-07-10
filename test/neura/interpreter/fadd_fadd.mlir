// RUN: neura-interpreter %s | FileCheck %s

// Test basic fused fadd operation: (2.5 + 1.5) + 3.0 = 7.0
func.func @test_fadd_fadd_basic() -> f32 {
  %a = arith.constant 2.5 : f32
  %b = arith.constant 1.5 : f32
  %c = arith.constant 3.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fadd_fadd:
  // CHECK: Calculation: (2.500000e+00 + 1.500000e+00) + 3.000000e+00 = 7.000000e+00
  // CHECK: Final result: value = 7.000000e+00, predicate = true
  // CHECK: [neura-interpreter] Output: 7.000000
  return %res : f32
}

// Test with negative numbers: (5.0 + (-2.0)) + (-1.0) = 2.0
func.func @test_fadd_fadd_negative() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant -2.0 : f32
  %c = arith.constant -1.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fadd_fadd:
  // CHECK: Calculation: (5.000000e+00 + -2.000000e+00) + -1.000000e+00 = 2.000000e+00
  // CHECK: [neura-interpreter] Output: 2.000000
  return %res : f32
}

// Test with zero: (0.0 + 4.0) + 6.0 = 10.0
func.func @test_fadd_fadd_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 4.0 : f32
  %c = arith.constant 6.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fadd_fadd:
  // CHECK: Calculation: (0.000000e+00 + 4.000000e+00) + 6.000000e+00 = 1.000000e+01
  // CHECK: [neura-interpreter] Output: 10.000000
  return %res : f32
}

// Test with valid predicate: (3.0 + 1.0) + 2.0 = 6.0
func.func @test_fadd_fadd_with_valid_predicate() -> f32 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 1.0 : f32
  %c = arith.constant 2.0 : f32
  %pred = arith.constant 1 : i1 
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fadd_fadd"(%a, %b, %c, %pred_f32) : (f32, f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fadd_fadd:
  // CHECK: Calculation: (3.000000e+00 + 1.000000e+00) + 2.000000e+00 = 6.000000e+00
  // CHECK: [neura-interpreter] Output: 6.000000
  return %res : f32
}

// Test with false predicate (should return 0)
func.func @test_fadd_fadd_with_invalid_predicate() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 20.0 : f32
  %c = arith.constant 30.0 : f32
  %pred = arith.constant 0 : i1  
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fadd_fadd"(%a, %b, %c, %pred_f32) : (f32, f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fadd_fadd:
  // CHECK: Predicate is false, result is 0
  // CHECK: Final result: value = 0.000000e+00, predicate = false
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}

// Test with invalid input predicate
func.func @test_fadd_fadd_with_invalid_input_predicate() -> f32 {
  %a = "neura.constant"() {value = 5.0 : f32, predicate = false} : () -> f32
  %b = arith.constant 3.0 : f32 
  %c = arith.constant 2.0 : f32
  %res = "neura.fadd_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fadd_fadd:
  // CHECK: Predicate is false, result is 0
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}