// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_gep_simple() -> i32 {
  %base = arith.constant 0 : i32
  %idx = arith.constant 2 : i32
  %gep = "neura.gep"(%base, %idx) { strides = [4] } : (i32, i32) -> i32

  %val = arith.constant 42 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()

  %loaded = "neura.load"(%gep) : (i32) -> i32
  // CHECK: [neura-interpreter]  └─ Final GEP result: base = 0, total offset = 8, final address = 8, [pred = 1]
  // CHECK: [neura-interpreter]  → Output: 42.000000
  return %loaded : i32
}

func.func @test_gep_2d() -> i32 {
  %base = arith.constant 0 : i32
  %idx0 = arith.constant 1 : i32
  %idx1 = arith.constant 3 : i32
  %gep = "neura.gep"(%base, %idx0, %idx1) { strides = [16, 4] } : (i32, i32, i32) -> i32
  // CHECK: [neura-interpreter]  └─ Final GEP result: base = 0, total offset = 28, final address = 28, [pred = 1]
  // CHECK: [neura-interpreter]  → Output: 99.000000  

  %val = arith.constant 99 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()

  %loaded = "neura.load"(%gep) : (i32) -> i32
  return %loaded : i32
}

func.func @test_gep_predicate() -> i32 {
  %base = arith.constant 0 : i32
  %idx0 = arith.constant 1 : i32
  %idx1 = arith.constant 3 : i32
  %pred = arith.constant 0 : i1

  %gep = "neura.gep"(%base, %idx0, %idx1, %pred) { strides = [16, 4] } : (i32, i32, i32, i1) -> i32
  // CHECK: [neura-interpreter]  └─ Final GEP result: base = 0, total offset = 28, final address = 28, [pred = 0]

  %val = arith.constant 77 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()

  %loaded = "neura.load"(%gep) : (i32) -> i32
  // CHECK: [neura-interpreter]  → Output: 0.000000
  return %loaded : i32
}