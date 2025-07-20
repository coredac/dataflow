// RUN: neura-interpreter %s | FileCheck %s

func.func @test_ctrl_mov_basic() {
  %a = "neura.reserve"() : () -> (i32)
  %const = arith.constant 42 : i32
  
  "neura.ctrl_mov"(%const, %a) : (i32, i32) -> ()
  
  // CHECK: [neura-interpreter]  → Output: (void)
  
  return
}

func.func @test_ctrl_mov_chained() {
  %a = "neura.reserve"() : () -> (i32)
  %b = "neura.reserve"() : () -> (i32)
  %const = arith.constant 10 : i32

  "neura.ctrl_mov"(%const, %a) : (i32, i32) -> ()

  "neura.ctrl_mov"(%a, %b) : (i32, i32) -> ()
  // CHECK: [neura-interpreter]  → Output: (void)

  return
}

func.func @test_ctrl_mov_vector() {
  %vec_reserve = "neura.reserve"() : () -> (vector<4xf32>)
  %vec_const = "neura.constant"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>} : () -> vector<4xf32>

  "neura.ctrl_mov"(%vec_const, %vec_reserve) : (vector<4xf32>, vector<4xf32>) -> ()
  // CHECK: [neura-interpreter]  → Output: (void)
  return
}