// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_not_nonzero() -> i32 {
  %a = arith.constant 42 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : i32
}

func.func @test_not_zero() -> i32 {
  %a = arith.constant 0 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 1.000000
  return %res : i32
}

func.func @test_not_negative() -> i32 {
  %a = arith.constant -1 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : i32
}

func.func @test_not_one() -> i32 {
  %a = arith.constant 1 : i32
  %res = "neura.not"(%a) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %res : i32
}
    