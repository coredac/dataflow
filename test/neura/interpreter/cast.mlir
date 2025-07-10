// RUN: neura-interpreter %s | FileCheck %s

// int -> float
func.func @test_cast_i2f() -> f32 {
  %a = arith.constant 42 : i32
  %res = "neura.cast"(%a) { cast_type = "i2f" } : (i32) -> f32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 4.200000e+01, predicate = 1
  // CHECK:  Cast type: i2f
  // CHECK:  Converting integer to float: 42 -> 4.200000e+01
  // CHECK:  Final result: value = 4.200000e+01, predicate = true
  // CHECK: [neura-interpreter] Output: 42.000000
  return %res : f32
}

// float -> int
func.func @test_cast_f2i() -> i32 {
  %a = arith.constant 3.14 : f32
  %res = "neura.cast"(%a) { cast_type = "f2i" } : (f32) -> i32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 3.140000e+00, predicate = 1
  // CHECK:  Cast type: f2i
  // CHECK:  Converting float to integer: 3.140000e+00 -> 3
  // CHECK:  Final result: value = 3.000000e+00, predicate = true
  // CHECK: [neura-interpreter] Output: 3.000000
  return %res : i32
}

// bool -> int
func.func @test_cast_bool2i() -> i32 {
  %b = arith.constant 1 : i1
  %res = "neura.cast"(%b) { cast_type = "bool2i" } : (i1) -> i32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 1.000000e+00, predicate = 1
  // CHECK:  Cast type: bool2i
  // CHECK:  Converting boolean to number: true -> 1.000000e+00
  // CHECK:  Final result: value = 1.000000e+00, predicate = true
  // CHECK: [neura-interpreter] Output: 1.000000
  return %res : i32
}

// bool -> float
func.func @test_cast_bool2f() -> f32 {
  %b = arith.constant 0 : i1
  %res = "neura.cast"(%b) { cast_type = "bool2f" } : (i1) -> f32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 0.000000e+00, predicate = 1
  // CHECK:  Cast type: bool2f
  // CHECK:  Converting boolean to number: false -> 0.000000e+00
  // CHECK:  Final result: value = 0.000000e+00, predicate = true
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : f32
}

// int -> bool
func.func @test_cast_i2bool() -> i1 {
  %a = arith.constant 100 : i32
  %res = "neura.cast"(%a) { cast_type = "i2bool" } : (i32) -> i1
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 1.000000e+02, predicate = 1
  // CHECK:  Cast type: i2bool
  // CHECK:  Converting number to boolean: 1.000000e+02 -> true (stored as 1.000000e+00)
  // CHECK:  Final result: value = 1.000000e+00, predicate = true
  // CHECK: [neura-interpreter] Output: 1.000000
  return %res : i1
}

// f2i with true predicate
func.func @test_cast_predicated() -> i32 {
  %val = arith.constant 5.5 : f32
  %pred = arith.constant 1 : i1
  %res = "neura.cast"(%val, %pred) { cast_type = "f2i" } : (f32, i1) -> i32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 5.500000e+00, predicate = 1
  // CHECK:  Cast type: f2i
  // CHECK:  Predicate operand: value = 1.000000e+00, predicate = 1
  // CHECK:  Converting float to integer: 5.500000e+00 -> 6
  // CHECK:  Final result: value = 6.000000e+00, predicate = true
  // CHECK: [neura-interpreter] Output: 6.000000
  return %res : i32
}

// f2i with false predicate
func.func @test_cast_predicate_false() -> i32 {
  %val = arith.constant 5.5 : f32
  %pred = arith.constant 0 : i1
  %res = "neura.cast"(%val, %pred) { cast_type = "f2i" } : (f32, i1) -> i32
  // CHECK: [neura-interpreter] Executing neura.cast:
  // CHECK:  Input: value = 5.500000e+00, predicate = 1
  // CHECK:  Cast type: f2i
  // CHECK:  Predicate operand: value = 0.000000e+00, predicate = 1
  // CHECK:  Predicate is false, result is 0
  // CHECK:  Final result: value = 0.000000e+00, predicate = false
  // CHECK: [neura-interpreter] Output: 0.000000
  return %res : i32
}