// RUN: mlir-neura-opt --assign-accelerator --lower-arith-to-neura %s | FileCheck %s --check-prefix=OPT

// CHECK-LABEL: func.func @test_cmpf(
// OPT: %{{.*}} = "neura.fcmp"
// OPT: cmpType = "ogt"
module {
  func.func @test_cmpf(%arg0: f32, %arg1: f32) -> i1 {
    %0 = arith.cmpf ogt, %arg0, %arg1 : f32
    return %0 : i1
  }
}