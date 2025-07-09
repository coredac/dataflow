// // RUN: mlir-neura-opt %s | FileCheck %s

// // ====== Conditional Branch Tests ======

// // Basic conditional branch with i1 condition
// func.func @test_cond_br_basic() {
//   %cond = arith.constant true
//   // CHECK: "neura.cond_br"
//   "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
// ^bb1:
//   return
// ^bb2:
//   return
// }

// // Conditional branch with predicate
// func.func @test_cond_br_with_predicate() {
//   %cond = arith.constant true
//   %pred = arith.constant 1 : i32
//   // CHECK: "neura.cond_br"
//   "neura.cond_br"(%cond, %pred) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 1, 0, 0>} : (i1, i32) -> ()
// ^bb1:
//   return
// ^bb2:
//   return
// }

// // // Conditional branch with arguments to true branch
// // func.func @test_cond_br_true_args() {
// //   %cond = arith.constant true
// //   %arg1 = arith.constant 42 : i32
// //   %arg2 = arith.constant 3.14 : f32
// //   // CHECK: "neura.cond_br"
// //   "neura.cond_br"(%cond, %arg1, %arg2) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 2, 0>} : (i1, i32, f32) -> ()
// // ^bb1(%a: i32, %b: f32):
// //   return
// // ^bb2:
// //   return
// // }

// // // Conditional branch with arguments to false branch
// // func.func @test_cond_br_false_args() {
// //   %cond = arith.constant false
// //   %arg1 = arith.constant 100 : i64
// //   // CHECK: "neura.cond_br"
// //   "neura.cond_br"(%cond, %arg1) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 1>} : (i1, i64) -> ()
// // ^bb1:
// //   return
// // ^bb2(%a: i64):
// //   return
// // }

// // // Conditional branch with both true and false arguments
// // func.func @test_cond_br_both_args() {
// //   %cond = arith.constant true
// //   %true_arg = arith.constant 1 : i32
// //   %false_arg = arith.constant 2 : i32
// //   // CHECK: "neura.cond_br"
// //   "neura.cond_br"(%cond, %true_arg, %false_arg) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 1, 1>} : (i1, i32, i32) -> ()
// // ^bb1(%a: i32):
// //   return
// // ^bb2(%b: i32):
// //   return
// // }

// // // Conditional branch with predicate and arguments
// // func.func @test_cond_br_predicate_and_args() {
// //   %cond = arith.constant false
// //   %pred = arith.constant 0 : i8
// //   %true_arg = arith.constant 10 : i32
// //   %false_arg = arith.constant 20 : i32
// //   // CHECK: "neura.cond_br"
// //   "neura.cond_br"(%cond, %pred, %true_arg, %false_arg) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 1, 1, 1>} : (i1, i8, i32, i32) -> ()
// // ^bb1(%a: i32):
// //   return
// // ^bb2(%b: i32):
// //   return
// // }

// // // Conditional branch with multiple arguments
// // func.func @test_cond_br_multiple_args() {
// //   %cond = arith.constant true
// //   %arg1 = arith.constant 1 : i32
// //   %arg2 = arith.constant 2.0 : f64
// //   %arg3 = arith.constant 3 : i16
// //   // CHECK: "neura.cond_br"
// //   "neura.cond_br"(%cond, %arg1, %arg2, %arg3) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 2, 1>} : (i1, i32, f64, i16) -> ()
// // ^bb1(%a: i32, %b: f64):
// //   return
// // ^bb2(%c: i16):
// //   return
// // }



// RUN: mlir-neura-opt %s | FileCheck %s

// ====== Conditional Branch Tests ======

// 1. Basic conditional branch with i1 condition (true case)
func.func @test_cond_br_true() {
  %cond = arith.constant 1 : i1 
  // CHECK: "neura.cond_br"
  "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
^bb1:
  // CHECK-NOT: Jump to false block
  return
^bb2:
  return
}

// 2. Basic conditional branch with i1 condition (false case)
func.func @test_cond_br_false() {
  %cond = arith.constant 0 : i1  
  // CHECK: "neura.cond_br"
  "neura.cond_br"(%cond) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()
^bb1:
  return
^bb2:
  // CHECK-NOT: Jump to true block
  return
}

// 3. Conditional branch with predicate (valid predicate)
func.func @test_cond_br_with_valid_predicate() {
  %cond = arith.constant 1 : i1
  %pred = arith.constant 1 : i32  
  // CHECK: "neura.cond_br"
  "neura.cond_br"(%cond, %pred) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 1, 0, 0>} : (i1, i32) -> ()
^bb1:
  // CHECK: Jump to true block
  return
^bb2:
  return
}

func.func @test_nested_cond_br() {
  %cond1 = arith.constant 1 : i1  
  "neura.cond_br"(%cond1) [^bb1, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^bb1:
  %cond2 = arith.constant 0 : i1 
  "neura.cond_br"(%cond2) [^bb3, ^bb4] {operandSegmentSizes = array<i32: 1, 0, 0, 0>} : (i1) -> ()

^bb2:
  return

^bb3:
  return

^bb4:
  return
}
// CHECK: Return value: 42