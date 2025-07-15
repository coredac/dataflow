// RUN: neura-interpreter %s | FileCheck %s

// (2.0 * 3.0) + 4.0 = 10.0
func.func @test_fmul_fadd_basic() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 3.0 : f32
  %c = arith.constant 4.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand C: value = 4.000000e+00, predicate = 1
  // CHECK-NEXT:   Calculation: (2.000000e+00 * 3.000000e+00) + 4.000000e+00 = 6.000000e+00 + 4.000000e+00 = 1.000000e+01
  // CHECK-NEXT:   Final result: value = 1.000000e+01, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 10.000000
  return %res : f32
}

// (5.0 * (-2.0)) + 12.0 = 2.0
func.func @test_fmul_fadd_negative() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant -2.0 : f32
  %c = arith.constant 12.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = -2.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand C: value = 1.200000e+01, predicate = 1
  // CHECK-NEXT:   Calculation: (5.000000e+00 * -2.000000e+00) + 1.200000e+01 = -1.000000e+01 + 1.200000e+01 = 2.000000e+00
  // CHECK-NEXT:   Final result: value = 2.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 2.000000
  return %res : f32
}

// (0.0 * 5.0) + 6.0 = 6.0
func.func @test_fmul_fadd_zero() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 5.0 : f32
  %c = arith.constant 6.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = 5.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand C: value = 6.000000e+00, predicate = 1
  // CHECK-NEXT:   Calculation: (0.000000e+00 * 5.000000e+00) + 6.000000e+00 = 0.000000e+00 + 6.000000e+00 = 6.000000e+00
  // CHECK-NEXT:   Final result: value = 6.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 6.000000
  return %res : f32
}

// (1.5 * 2.0) + 3.5 = 6.5
func.func @test_fmul_fadd_decimal() -> f32 {
  %a = arith.constant 1.5 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.constant 3.5 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 1.500000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand C: value = 3.500000e+00, predicate = 1
  // CHECK-NEXT:   Calculation: (1.500000e+00 * 2.000000e+00) + 3.500000e+00 = 3.000000e+00 + 3.500000e+00 = 6.500000e+00
  // CHECK-NEXT:   Final result: value = 6.500000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 6.500000
  return %res : f32
}

// (4.0 * 2.0) + 1.0 = 9.0
func.func @test_fmul_fadd_with_valid_predicate() -> f32 {
  %a = arith.constant 4.0 : f32
  %b = arith.constant 2.0 : f32
  %c = arith.constant 1.0 : f32
  %pred = arith.constant 1 : i1  // predicate 为 true
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fmul_fadd"(%a, %b, %c, %pred_f32) : (f32, f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK-NEXT:   Input: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   Cast type: bool2f
  // CHECK-NEXT:   Converting boolean to number: true -> 1.000000e+00
  // CHECK-NEXT:   Final result: value = 1.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 4.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand C: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   Predicate: value = 1.000000e+00, predicate = 1
  // CHECK-NEXT:   Calculation: (4.000000e+00 * 2.000000e+00) + 1.000000e+00 = 8.000000e+00 + 1.000000e+00 = 9.000000e+00
  // CHECK-NEXT:   Final result: value = 9.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Output: 9.000000
  return %res : f32
}

func.func @test_fmul_fadd_with_invalid_predicate() -> f32 {
  %a = arith.constant 3.0 : f32
  %b = arith.constant 3.0 : f32
  %c = arith.constant 3.0 : f32
  %pred = arith.constant 0 : i1
  %pred_f32 = "neura.cast"(%pred) {cast_type = "bool2f"} : (i1) -> f32
  %res = "neura.fmul_fadd"(%a, %b, %c, %pred_f32) : (f32, f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK-NEXT:   Input: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   Cast type: bool2f
  // CHECK-NEXT:   Converting boolean to number: false -> 0.000000e+00
  // CHECK-NEXT:   Final result: value = 0.000000e+00, predicate = true
  // CHECK-NEXT: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand C: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Predicate: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   Predicate is false, result is 0
  // CHECK-NEXT:   Final result: value = 0.000000e+00, predicate = false
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %res : f32
}

func.func @test_fmul_fadd_with_invalid_input_predicate() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = "neura.constant"() {value = 5.0 : f32, predicate = false} : () -> f32
  %c = arith.constant 3.0 : f32
  %res = "neura.fmul_fadd"(%a, %b, %c) : (f32, f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.constant:
  // CHECK-NEXT: [neura-interpreter] Executing neura.fmul_fadd:
  // CHECK-NEXT:   Operand A: value = 2.000000e+00, predicate = 1
  // CHECK-NEXT:   Operand B: value = 5.000000e+00, predicate = 0
  // CHECK-NEXT:   Operand C: value = 3.000000e+00, predicate = 1
  // CHECK-NEXT:   Predicate is false, result is 0
  // CHECK-NEXT:   Final result: value = 0.000000e+00, predicate = false
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %res : f32
}