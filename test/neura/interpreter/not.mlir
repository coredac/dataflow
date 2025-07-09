// RUN: neura-interpreter %s | FileCheck %s

func.func @test_not_basic() -> i32 {
  %a = arith.constant 42 : i32
  %res = "neura.not"(%a) : (i32) -> i32

  // CHECK: neura.not

  return %res : i32
}

func.func @test_not_zero() -> i32 {
  %a = arith.constant 0 : i32
  %res = "neura.not"(%a) : (i32) -> i32

  // CHECK: neura.not

  return %res : i32
}

func.func @test_not_one() -> i32 {
  %a = arith.constant 1 : i32
  %res = "neura.not"(%a) : (i32) -> i32

  // CHECK: neura.not

  return %res : i32
}