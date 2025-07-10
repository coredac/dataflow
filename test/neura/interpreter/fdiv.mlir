// RUN: neura-interpreter %s | FileCheck %s

func.func @test_fdiv_positive() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 1.000000e+01, predicate = 1
  // CHECK:  RHS: value = 2.000000e+00, predicate = 1
  // CHECK:  Result: value = 5.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 5.000000
  return %res : f32
}

func.func @test_fdiv_negative_dividend() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = -1.000000e+01, predicate = 1
  // CHECK:  RHS: value = 2.000000e+00, predicate = 1
  // CHECK:  Result: value = -5.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: -5.000000
  return %res : f32
}

func.func @test_fdiv_negative_divisor() -> f32 {
  %a = arith.constant 10.0 : f32
  %b = arith.constant -2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 1.000000e+01, predicate = 1
  // CHECK:  RHS: value = -2.000000e+00, predicate = 1
  // CHECK:  Result: value = -5.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: -5.000000
  return %res : f32
}

func.func @test_fdiv_two_negatives() -> f32 {
  %a = arith.constant -10.0 : f32
  %b = arith.constant -2.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = -1.000000e+01, predicate = 1
  // CHECK:  RHS: value = -2.000000e+00, predicate = 1
  // CHECK:  Result: value = 5.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 5.000000
  return %res : f32
}

func.func @test_fdiv_by_zero() -> f32 {
  %a = arith.constant 5.0 : f32
  %b = arith.constant 0.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 5.000000e+00, predicate = 1
  // CHECK:  RHS: value = 0.000000e+00, predicate = 1
  // CHECK:  Warning: Division by zero, result is NaN
  // CHECK:  Result: value = nan, predicate = 1
  // CHECK: [neura-interpreter] Output: nan
  return %res : f32
}

func.func @test_fdiv_zero_dividend() -> f32 {
  %a = arith.constant 0.0 : f32
  %b = arith.constant 5.0 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 0.000000e+00, predicate = 1
  // CHECK:  RHS: value = 5.000000e+00, predicate = 1
  // CHECK:  Result: value = 0.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}

func.func @test_fdiv_with_predicate_true() -> f32 {
  %a = arith.constant 15.0 : f32
  %b = arith.constant 3.0 : f32
  %pred = arith.constant 1 : i32
  %res = "neura.fdiv"(%a, %b, %pred) : (f32, f32, i32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 1.500000e+01, predicate = 1
  // CHECK:  RHS: value = 3.000000e+00, predicate = 1
  // CHECK:  Predicate: value = 1.000000e+00, predicate = 1
  // CHECK:  Result: value = 5.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 5.000000
  return %res : f32
}

func.func @test_fdiv_with_predicate_false() -> f32 {
  %a = arith.constant 20.0 : f32
  %b = arith.constant 4.0 : f32
  %pred = arith.constant 0 : i32
  %res = "neura.fdiv"(%a, %b, %pred) : (f32, f32, i32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 2.000000e+01, predicate = 1
  // CHECK:  RHS: value = 4.000000e+00, predicate = 1
  // CHECK:  Predicate: value = 0.000000e+00, predicate = 1
  // CHECK:  Result: value = 0.000000e+00, predicate = 0
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}

func.func @test_fdiv_f64() -> f64 {
  %a = arith.constant 10.5 : f64
  %b = arith.constant 2.5 : f64
  %res = "neura.fdiv"(%a, %b) : (f64, f64) -> f64
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 1.050000e+01, predicate = 1
  // CHECK:  RHS: value = 2.500000e+00, predicate = 1
  // CHECK:  Result: value = 4.200000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 4.200000
  return %res : f64
}

func.func @test_fdiv_decimal() -> f32 {
  %a = arith.constant 2.0 : f32
  %b = arith.constant 0.5 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 2.000000e+00, predicate = 1
  // CHECK:  RHS: value = 5.000000e-01, predicate = 1
  // CHECK:  Result: value = 4.000000e+00, predicate = 1
  // CHECK: [neura-interpreter] Output: 4.000000
  return %res : f32
}

func.func @test_fdiv_large_numbers() -> f32 {
  %a = arith.constant 1.0e20 : f32
  %b = arith.constant 1.0e10 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 1.000000e+20, predicate = 1
  // CHECK:  RHS: value = 1.000000e+10, predicate = 1
  // CHECK:  Result: value = 1.000000e+10, predicate = 1
  // CHECK: [neura-interpreter] Output: 10000000000.000000
  return %res : f32
}

func.func @test_fdiv_near_zero() -> f32 {
  %a = arith.constant 1.0e-20 : f32
  %b = arith.constant 1.0e-10 : f32
  %res = "neura.fdiv"(%a, %b) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Executing neura.fdiv:
  // CHECK:  LHS: value = 1.000000e-20, predicate = 1
  // CHECK:  RHS: value = 1.000000e-10, predicate = 1
  // CHECK:  Result: value = 9.999999e-11, predicate = 1
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}