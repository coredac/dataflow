// RUN: mlir-neura-opt --lower-llvm-to-neura --insert-mov %s | FileCheck %s

func.func @test(%a: f32) -> f32 {
  %b = llvm.mlir.constant(2.0 : f32) : f32
  %res = llvm.fadd %a, %b : f32
  // CHECK: [[LHS:%.*]] = neura.mov %{{.*}} : f32 -> f32
  // CHECK: [[RHS:%.*]] = neura.mov %{{.*}} : f32 -> f32
  // CHECK: [[RES:%.*]] = neura.add [[LHS]], [[RHS]] : f32
  return %res : f32
}
