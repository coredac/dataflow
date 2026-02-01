// RUN: mkdir -p %S/Output
// RUN: cd %S && python build_modular_mlir.py --output %S/Output/nerf_modular_3funcs.mlir
// RUN: mlir-neura-opt %S/Output/nerf_modular_3funcs.mlir --assign-task-target -o %S/Output/nerf_with_target.mlir
// RUN: mlir-neura-opt %S/Output/nerf_modular_3funcs.mlir --assign-task-target | FileCheck %s

// Test AssignTaskTarget pass on NeRF modular functions
// This test verifies the complete workflow:
//   1. Generate modular MLIR from PyTorch NeRF components using build_modular_mlir.py
//   2. Run AssignTaskTarget pass to assign hardware targets to functions
//   3. Verify that targets are correctly assigned based on function names:
//      - ray_sampler_func -> CPU (sampling operations)
//      - hash_encoder_func -> DOE (encoding operations)
//      - nerf_mlp_func -> CGRA (neural network inference)
//      - nerf_forward -> CPU (top-level coordinator)

// CHECK-LABEL: func.func @ray_sampler_func
// CHECK-SAME: attributes {target.device = "cpu"}

// CHECK-LABEL: func.func @hash_encoder_func
// CHECK-SAME: attributes {target.device = "doe"}

// CHECK-LABEL: func.func @nerf_mlp_func
// CHECK-SAME: attributes {target.device = "cgra"}

// CHECK-LABEL: func.func @nerf_forward
// CHECK-SAME: attributes {target.device = "cpu"}
