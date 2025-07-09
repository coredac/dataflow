// RUN: neura-interpreter %s | FileCheck %s

// int -> float
func.func @test_cast_i2f() -> f32 {
  %a = arith.constant 42 : i32
  %res = "neura.cast"(%a) { cast_type = "i2f" } : (i32) -> f32
  // CHECK: Cast [i2f] input: 42 -> result: 42.0
  return %res : f32
}

// float -> int
func.func @test_cast_f2i() -> i32 {
  %a = arith.constant 3.14 : f32
  %res = "neura.cast"(%a) { cast_type = "f2i" } : (f32) -> i32
  // CHECK: Cast [f2i] input: 3.14 -> result: 3
  return %res : i32
}

// bool -> int
func.func @test_cast_bool2i() -> i32 {
  %b = arith.constant 1 : i1
  %res = "neura.cast"(%b) { cast_type = "bool2i" } : (i1) -> i32
  // CHECK: Cast [bool2i] input: 1 -> result: 1
  return %res : i32
}

// bool -> float
func.func @test_cast_bool2f() -> f32 {
  %b = arith.constant 0 : i1
  %res = "neura.cast"(%b) { cast_type = "bool2f" } : (i1) -> f32
  // CHECK: Cast [bool2f] input: 0 -> result: 0.0
  return %res : f32
}

// int -> bool
func.func @test_cast_i2bool() -> i1 {
  %a = arith.constant 100 : i32
  %res = "neura.cast"(%a) { cast_type = "i2bool" } : (i32) -> i1
  // CHECK: Cast [i2bool] input: 100 -> result: 1
  return %res : i1
}

// f2i
func.func @test_cast_predicated() -> i32 {
  %val = arith.constant 5.5 : f32
  %pred = arith.constant 1 : i1
  %res = "neura.cast"(%val, %pred) { cast_type = "f2i" } : (f32, i1) -> i32
  // CHECK: Cast [f2i] input: 5.5 -> result: 6
  return %res : i32
}

func.func @test_cast_predicate_false() -> i32 {
  %val = arith.constant 5.5 : f32
  %pred = arith.constant 0 : i1
  %res = "neura.cast"(%val, %pred) { cast_type = "f2i" } : (f32, i1) -> i32
  // CHECK: Cast [f2i] input: 5.5 -> result: 6
  return %res : i32
}
