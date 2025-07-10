// RUN: neura-interpreter %s | FileCheck %s

func.func @test_gep_simple() -> i32 {
  %base = arith.constant 0 : i32
  %idx = arith.constant 2 : i32
  %gep = "neura.gep"(%base, %idx) { strides = [4] } : (i32, i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.gep:
  // CHECK-NEXT:   Base address: value = 0, predicate = 1
  // CHECK-NEXT:   Index 0: value = 2, stride = 4, cumulative offset = 8
  // CHECK-NEXT:   Final GEP result: base = 0, total offset = 8, final address = 8, predicate = true

  %val = arith.constant 42 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()
  // CHECK: [neura-interpreter] Executing neura.store:
  // CHECK-NEXT:   Store [addr = 8] <= val = 4.200000e+01 (predicate=true)

  %loaded = "neura.load"(%gep) : (i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.load:
  // CHECK-NEXT:   Load  [addr = 8] => val = 4.200000e+01 (predicate=true)
  // CHECK-NEXT: [neura-interpreter] Output: 42.000000
  return %loaded : i32
}

func.func @test_gep_2d() -> i32 {
  %base = arith.constant 0 : i32
  %idx0 = arith.constant 1 : i32
  %idx1 = arith.constant 3 : i32
  %gep = "neura.gep"(%base, %idx0, %idx1) { strides = [16, 4] } : (i32, i32, i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.gep:
  // CHECK-NEXT:   Base address: value = 0, predicate = 1
  // CHECK-NEXT:   Index 0: value = 1, stride = 16, cumulative offset = 16
  // CHECK-NEXT:   Index 1: value = 3, stride = 4, cumulative offset = 28
  // CHECK-NEXT:   Final GEP result: base = 0, total offset = 28, final address = 28, predicate = true

  %val = arith.constant 99 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()
  // CHECK: [neura-interpreter] Executing neura.store:
  // CHECK-NEXT:   Store [addr = 28] <= val = 9.900000e+01 (predicate=true)

  %loaded = "neura.load"(%gep) : (i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.load:
  // CHECK-NEXT:   Load  [addr = 28] => val = 9.900000e+01 (predicate=true)
  // CHECK-NEXT: [neura-interpreter] Output: 99.000000
  return %loaded : i32
}

func.func @test_gep_predicate() -> i32 {
  %base = arith.constant 0 : i32
  %idx0 = arith.constant 1 : i32
  %idx1 = arith.constant 3 : i32
  %pred = arith.constant 0 : i1

  %gep = "neura.gep"(%base, %idx0, %idx1, %pred) { strides = [16, 4] } : (i32, i32, i32, i1) -> i32
  // CHECK: [neura-interpreter] Executing neura.gep:
  // CHECK-NEXT:   Base address: value = 0, predicate = 1
  // CHECK-NEXT:   Index 0: value = 1, stride = 16, cumulative offset = 16
  // CHECK-NEXT:   Index 1: value = 3, stride = 4, cumulative offset = 28
  // CHECK-NEXT:   Predicate operand: value = 0.000000e+00, predicate = 1
  // CHECK-NEXT:   Final GEP result: base = 0, total offset = 28, final address = 28, predicate = false

  %val = arith.constant 77 : i32
  "neura.store"(%val, %gep) : (i32, i32) -> ()
  // CHECK: [neura-interpreter] Executing neura.store:
  // CHECK-NEXT:   Store [addr = 28] skipped due to predicate=false

  %loaded = "neura.load"(%gep) : (i32) -> i32
  // CHECK: [neura-interpreter] Executing neura.load:
  // CHECK-NEXT:   Load  [addr = 28] => val = 0.000000e+00 (predicate=false)
  // CHECK-NEXT: [neura-interpreter] Output: 0.000000
  return %loaded : i32
}