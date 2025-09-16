// RUN: neura-interpreter %s --verbose | FileCheck %s

// Test basic subtraction with positive result
func.func @test_sub_positive() -> i32 {
  %a = arith.constant 200 : i32
  %b = arith.constant 50 : i32
  %res = "neura.sub"(%a, %b) : (i32, i32) -> i32

  // CHECK: [neura-interpreter]  → Output: 150.000000
  return %res : i32
}

// Test subtraction with negative result
func.func @test_sub_negative() -> i32 {
  %a = arith.constant 50 : i32
  %b = arith.constant 200 : i32
  %res = "neura.sub"(%a, %b) : (i32, i32) -> i32

  // CHECK: [neura-interpreter]  → Output: -150.000000
  return %res : i32
}

// Remove tests with predicate input because we plan to remove the predicate attribute in
// https://github.com/coredac/dataflow/issues/116

// Test subtraction with predicate=true
// func.func @test_sub_with_predicate_true() -> i32 {
//   %a = arith.constant 300 : i32
//   %b = arith.constant 100 : i32
//   %pred = arith.constant 1 : i32  
//   %res = "neura.sub"(%a, %b, %pred) : (i32, i32, i32) -> i32

//   return %res : i32
// }


// Test subtraction with predicate=false
// func.func @test_sub_with_predicate_false() -> i32 {
//   %a = arith.constant 500 : i32
//   %b = arith.constant 200 : i32
//   %pred = arith.constant 0 : i32  
//   %res = "neura.sub"(%a, %b, %pred) : (i32, i32, i32) -> i32
//   return %res : i32
// }