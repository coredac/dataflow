// RUN: neura-interpreter %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: Basic Phi node with control flow
//===----------------------------------------------------------------------===//
func.func @test_phi_ctrlflow() -> f32 {
  %init = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32

  %v = "neura.reserve"() : () -> (f32)
  "neura.ctrl_mov"(%one, %v) : (f32, f32) -> ()

  %cond = arith.constant 0 : i1
  "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^bb1:
  "neura.ctrl_mov"(%init, %v) : (f32, f32) -> ()
  "neura.br"() [^merge] {operandSegmentSizes = array<i32: 0>} : () -> ()

^bb2:
  "neura.br"() [^merge] {operandSegmentSizes = array<i32: 0>} : () -> ()

^merge:
  %phi = "neura.phi"(%init, %v) : (f32, f32) -> f32
  // CHECK: [neura-interpreter] Output: 0.000000
  return %phi : f32
}

//===----------------------------------------------------------------------===//
// Test 2: Phi node in loop structure
//===----------------------------------------------------------------------===//
func.func @test_loop_phi() -> f32 {
  %init = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %limit = arith.constant 3.0 : f32

  %v = "neura.reserve"() : () -> (f32)
  "neura.ctrl_mov"(%init, %v) : (f32, f32) -> ()

  "neura.br"() [^loop_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^loop_head:
  %i = "neura.phi"(%v, %init) : (f32, f32) -> f32

  %cond = "neura.fcmp"(%i, %limit) {cmpType = "lt"} : (f32, f32) -> i1

  "neura.cond_br"(%cond) [^loop_body, ^loop_exit] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^loop_body:
  %i_next = "neura.fadd"(%i, %one) : (f32, f32) -> f32
  "neura.ctrl_mov"(%i_next, %v) : (f32, f32) -> ()
  "neura.br"() [^loop_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^loop_exit:
  // CHECK: [neura-interpreter] Output: 3.000000
  return %i : f32
}

//===----------------------------------------------------------------------===//
// Test 3: Phi node with multiple predecessors
//===----------------------------------------------------------------------===//
func.func @test_phi_multi_preds() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cond = arith.constant 0 : i1  
  "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^bb1:
  "neura.br"() [^merge] : () -> ()

^bb2:
  "neura.br"() [^bb3] : () -> ()

^bb3:
  "neura.br"() [^merge] : () -> ()

^extra:
  "neura.br"() [^merge] : () -> ()

^merge:
  %val = "neura.phi"(%c0, %c1, %c2) : (i32, i32, i32) -> i32
  // CHECK: [neura-interpreter] Output: 1.000000
  return %val : i32
}

//===----------------------------------------------------------------------===//
// Test 4: Nested loops with phi nodes
//===----------------------------------------------------------------------===//
func.func @test_small_nested_loops() -> f32 {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32

  %outer_i = "neura.reserve"() : () -> f32
  "neura.ctrl_mov"(%zero, %outer_i) : (f32, f32) -> ()

  %outer_sum = "neura.reserve"() : () -> f32
  "neura.ctrl_mov"(%zero, %outer_sum) : (f32, f32) -> ()
  "neura.br"() [^outer_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^outer_head:
  %i = "neura.phi"(%outer_i, %zero) : (f32, f32) -> f32
  %outer_cond = "neura.fcmp"(%i, %two) {cmpType = "lt"} : (f32, f32) -> i1
  "neura.cond_br"(%outer_cond) [^outer_body, ^outer_exit] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^outer_body:
  %inner_j = "neura.reserve"() : () -> f32
  "neura.ctrl_mov"(%zero, %inner_j) : (f32, f32) -> ()
  %inner_sum = "neura.reserve"() : () -> f32
  "neura.ctrl_mov"(%zero, %inner_sum) : (f32, f32) -> ()
  "neura.br"() [^inner_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^inner_head:
  %j = "neura.phi"(%inner_j, %zero) : (f32, f32) -> f32
  %inner_cond = "neura.fcmp"(%j, %two) {cmpType = "lt"} : (f32, f32) -> i1
  "neura.cond_br"(%inner_cond) [^inner_body, ^inner_exit] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^inner_body:
  %i_mul_two = "neura.fmul"(%i, %two) : (f32, f32) -> f32
  %current_val = "neura.fadd"(%i_mul_two, %j) : (f32, f32) -> f32
  %new_inner_sum = "neura.fadd"(%inner_sum, %current_val) : (f32, f32) -> f32
  "neura.ctrl_mov"(%new_inner_sum, %inner_sum) : (f32, f32) -> ()
  %j_next = "neura.fadd"(%j, %one) : (f32, f32) -> f32
  "neura.ctrl_mov"(%j_next, %inner_j) : (f32, f32) -> ()
  "neura.br"() [^inner_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^inner_exit:
  %new_outer_sum = "neura.fadd"(%outer_sum, %inner_sum) : (f32, f32) -> f32
  "neura.ctrl_mov"(%new_outer_sum, %outer_sum) : (f32, f32) -> ()
  %i_next = "neura.fadd"(%i, %one) : (f32, f32) -> f32
  "neura.ctrl_mov"(%i_next, %outer_i) : (f32, f32) -> ()
  "neura.br"() [^outer_head] {operandSegmentSizes = array<i32: 0>} : () -> ()

^outer_exit:
  // CHECK: [neura-interpreter] Output: 6.000000
  return %outer_sum : f32
}