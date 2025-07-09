// RUN: neura-interpreter %s | FileCheck %s

func.func @test_sub_positive() -> i32 {
  %a = arith.constant 200 : i32
  %b = arith.constant 50 : i32
  %res = "neura.sub"(%a, %b) : (i32, i32) -> i32

  // CHECK: neura.sub

  return %res : i32
}

func.func @test_sub_negative() -> i32 {
  %a = arith.constant 50 : i32
  %b = arith.constant 200 : i32
  %res = "neura.sub"(%a, %b) : (i32, i32) -> i32

  // CHECK: neura.sub

  return %res : i32
}

func.func @test_sub_with_predicate_true() -> i32 {
  %a = arith.constant 300 : i32
  %b = arith.constant 100 : i32
  %pred = arith.constant 1 : i32  // 非零表示 predicate = true

  %res = "neura.sub"(%a, %b, %pred) : (i32, i32, i32) -> i32

  // CHECK: neura.sub

  return %res : i32
}

func.func @test_sub_with_predicate_false() -> i32 {
  %a = arith.constant 500 : i32
  %b = arith.constant 200 : i32
  %pred = arith.constant 0 : i32  // 0 表示 predicate = false

  %res = "neura.sub"(%a, %b, %pred) : (i32, i32, i32) -> i32

  // CHECK: neura.sub

  return %res : i32
}