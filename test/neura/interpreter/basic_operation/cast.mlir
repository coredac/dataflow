// RUN: neura-interpreter %s --verbose | FileCheck %s

// int -> float
func.func @test_cast_i2f() -> f32 {
  %a = arith.constant 42 : i32
  %res = "neura.cast"(%a) { cast_type = "i2f" } : (i32) -> f32
  // CHECK: [neura-interpreter]  → Output: 42.000000
  return %res : f32
}

// float -> int
func.func @test_cast_f2i() -> i32 {
  %a = arith.constant 3.14 : f32
  %res = "neura.cast"(%a) { cast_type = "f2i" } : (f32) -> i32
  // CHECK: [neura-interpreter]  → Output: 3.000000
  return %res : i32
}

// bool -> int
func.func @test_cast_bool2i() -> i32 {
  %b = arith.constant 1 : i1
  %res = "neura.cast"(%b) { cast_type = "bool2i" } : (i1) -> i32
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : i32
}

// bool -> float
func.func @test_cast_bool2f() -> f32 {
  %b = arith.constant 0 : i1
  %res = "neura.cast"(%b) { cast_type = "bool2f" } : (i1) -> f32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : f32
}

// int -> bool
func.func @test_cast_i2bool() -> i1 {
  %a = arith.constant 100 : i32
  %res = "neura.cast"(%a) { cast_type = "i2bool" } : (i32) -> i1
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : i1
}

// f2i with true predicate
func.func @test_cast_embed_predicated() -> i32 {
  %val = "neura.constant"() {value = 5.5 : f32} : () -> f32
  %res = "neura.cast"(%val) { cast_type = "f2i" } : (f32) -> i32
  // CHECK: [neura-interpreter]  → Output: 6.000000
  return %res : i32
}