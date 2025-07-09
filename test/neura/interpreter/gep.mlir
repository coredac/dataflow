// RUN: neura-interpreter %s | FileCheck %s

func.func @test_gep_simple() -> i32 {
  %base = arith.constant 0 : i32              // 基址0
  %idx = arith.constant 2 : i32               // 索引2
  %gep = "neura.gep"(%base, %idx) 
         { strides = [4] } : (i32, i32) -> i32  // 计算地址 = 0 + 2*4 = 8

  // 在地址8存一个42
  %val = arith.constant 42 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()

  // 从地址8读回来
  %loaded = "neura.load"(%gep) : (i32) -> i32

  // CHECK: GEP base [0] + offset [8] = 8
  // CHECK: store value: 42 at addr 8
  // CHECK: load value: 42 from addr 8
  return %loaded : i32
}

func.func @test_gep_2d() -> i32 {
  %base = arith.constant 0 : i32           // 基址0
  %idx0 = arith.constant 1 : i32           // 第一维索引1
  %idx1 = arith.constant 3 : i32           // 第二维索引3
  %gep = "neura.gep"(%base, %idx0, %idx1) 
         { strides = [16, 4] } : (i32, i32, i32) -> i32  // 计算地址 = 28

  // 在地址28存一个99
  %val = arith.constant 99 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()

  // 从地址28读回来
  %loaded = "neura.load"(%gep) : (i32) -> i32

  // CHECK: GEP base [0] + offset [28] = 28
  // CHECK: store value: 99 at addr 28
  // CHECK: load value: 99 from addr 28
  return %loaded : i32
}

func.func @test_gep_predicate() -> i32 {
  %base = arith.constant 0 : i32
  %idx0 = arith.constant 1 : i32
  %idx1 = arith.constant 3 : i32
  %pred = arith.constant 0 : i1

  %gep = "neura.gep"(%base, %idx0, %idx1, %pred) { strides = [16, 4] } : (i32, i32, i32, i1) -> i32

  %val = arith.constant 77 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()

  %loaded = "neura.load"(%gep) : (i32) -> i32

  return %loaded : i32
}
