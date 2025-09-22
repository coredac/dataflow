// RUN: neura-interpreter %s --verbose | FileCheck %s

func.func @test_gep_simple() -> i32 {
  %base = arith.constant 0 : i32
  %idx = arith.constant 2 : i32
  %gep = "neura.gep"(%base, %idx) { strides = [4] } : (i32, i32) -> i32
  // CHECK: [neura-interpreter]  └─ Final GEP result: base = 0, total offset = 8, final address = 8, [pred = 1]
  
  return %gep : i32
}