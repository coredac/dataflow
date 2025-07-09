// RUN: mlir-neura-opt %s | FileCheck %s

func.func @test_ctrl_mov_basic() {
  %a = "neura.reserve"() : () -> (i32)
  %const = arith.constant 42 : i32
  "neura.ctrl_mov"(%const, %a) : (i32, i32) -> ()
  
  // CHECK: neura.reserve: created placeholder
  // CHECK: neura.ctrl_mov: Source: %const
  // CHECK:   Value: 42
  // CHECK:   Target: %a
  // CHECK:   Updated target placeholder:
  // CHECK:     New value: 42
  
  return
}

// RUN: mlir-neura-opt %s | FileCheck %s

func.func @test_ctrl_mov_chained() {
  %a = "neura.reserve"() : () -> (i32)
  %b = "neura.reserve"() : () -> (i32)
  %const = arith.constant 10 : i32
  
  "neura.ctrl_mov"(%const, %a) : (i32, i32) -> ()
  "neura.ctrl_mov"(%a, %b) : (i32, i32) -> ()
  
  // CHECK: neura.ctrl_mov: Updated target placeholder:
  // CHECK:   New value: 10
  // CHECK: neura.ctrl_mov: Updated target placeholder:
  // CHECK:   New value: 10
  
  return
}

// RUN: mlir-neura-opt %s | FileCheck %s

func.func @test_ctrl_mov_vector() {
  %vec_reserve = "neura.reserve"() : () -> (vector<4xf32>)
  %vec_const = "neura.constant"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>} : () -> vector<4xf32>
  "neura.ctrl_mov"(%vec_const, %vec_reserve) : (vector<4xf32>, vector<4xf32>) -> ()
  
  // CHECK: neura.ctrl_mov: Source: %vec_const
  // CHECK:   Value: [1.000000, 2.000000, 3.000000, 4.000000]
  // CHECK:   Target: %vec_reserve
  // CHECK:   Updated target placeholder:
  // CHECK:     New value: [1.000000, 2.000000, 3.000000, 4.000000]
  
  return
}