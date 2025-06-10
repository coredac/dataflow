// RUN: mlir-neura-opt --assign-accelerator --lower-llvm-to-neura --insert-data-mov %s | FileCheck %s

func.func @test(%a: f32) -> f32 {
  %b = llvm.mlir.constant(2.0 : f32) : f32
  %res = llvm.fadd %a, %b : f32
  // CHECK: [[LHS:%.*]] = "neura.data_mov"(%{{.*}}) : (f32) -> f32
  // CHECK: [[RHS:%.*]] = "neura.data_mov"(%{{.*}}) : (f32) -> f32
  // CHECK: [[RES:%.*]] = "neura.fadd"([[LHS]], [[RHS]])
  return %res : f32
}
