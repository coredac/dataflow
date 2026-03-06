// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: -o %t.serialized.mlir
// RUN: FileCheck %s --input-file=%t.serialized.mlir --check-prefixes=SERIALIZED

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --convert-affine-to-taskflow \
// RUN: -o %t.taskflow.mlir
// RUN: FileCheck %s --input-file=%t.taskflow.mlir --check-prefixes=TASKFLOW

// RUN: mlir-neura-opt %s --affine-loop-tree-serialization \
// RUN: --affine-loop-perfection \
// RUN: --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --classify-counters \
// RUN: --convert-taskflow-to-neura \
// RUN: --lower-affine \
// RUN: --convert-scf-to-cf \
// RUN: --convert-cf-to-llvm \
// RUN: --assign-accelerator \
// RUN: --lower-memref-to-neura \
// RUN: --lower-arith-to-neura \
// RUN: --lower-builtin-to-neura \
// RUN: --lower-llvm-to-neura \
// RUN: --promote-input-arg-to-const \
// RUN: --fold-constant \
// RUN: --canonicalize-return \
// RUN: --canonicalize-live-in \
// RUN: --leverage-predicated-value \
// RUN: --transform-ctrl-to-data-flow \
// RUN: --fold-constant \
// RUN: '--resource-aware-task-optimization' \
// RUN: --architecture-spec=%S/../../../arch_spec/architecture_with_counter.yaml \
// RUN: -o %t.resopt.mlir
// RUN: FileCheck %s --input-file=%t.resopt.mlir --check-prefixes=RESOPT

module {
  // Example: Stereo image disparity preprocessing — a real computer vision kernel.
  //
  // This models a common pattern in stereo vision pipelines where for each pixel
  // pair from left/right cameras, we compute multiple cost metrics (SAD variants)
  // plus gradient-based features. The heavy per-pixel compute makes res_mii > 1
  // on a single 4×4 CGRA (16 tiles), forcing the balance pass to allocate
  // multiple CGRAs.
  //
  // Task 0 (heavy): Stereo cost computation — per pixel computes:
  //   - Left/right scale+bias normalization (6 channels: R,G,B × 2 views)
  //   - Channel-wise absolute differences
  //   - Weighted sum of absolute differences
  //   - Gradient features (horizontal differences)
  //   Total: ~40+ materialized Neura ops → res_mii=3 on 16 tiles.
  //   On 32 tiles (2 CGRAs), res_mii drops to 2, enabling II reduction.
  //
  // Task 1 (light): Simple post-processing (few ops, res_mii=1).
  func.func @stereo_cost_computation(
      %L_R: memref<64xf32>, %L_G: memref<64xf32>, %L_B: memref<64xf32>,
      %R_R: memref<64xf32>, %R_G: memref<64xf32>, %R_B: memref<64xf32>,
      %cost: memref<64xf32>, %grad: memref<64xf32>,
      %w1: f32, %w2: f32, %w3: f32,
      %scale: f32, %bias: f32,
      %aux_in: memref<64xf32>, %aux_out: memref<64xf32>) {

    // Task 0: Stereo matching cost with multi-feature extraction
    affine.for %i = 0 to 64 {
      // Load left view RGB
      %lr = affine.load %L_R[%i] : memref<64xf32>
      %lg = affine.load %L_G[%i] : memref<64xf32>
      %lb = affine.load %L_B[%i] : memref<64xf32>

      // Load right view RGB
      %rr = affine.load %R_R[%i] : memref<64xf32>
      %rg = affine.load %R_G[%i] : memref<64xf32>
      %rb = affine.load %R_B[%i] : memref<64xf32>

      // Normalize left: l_ch = L_ch * scale + bias  (6 ops: 3 fmul + 3 fadd)
      %lr_s = arith.mulf %lr, %scale : f32
      %lr_n = arith.addf %lr_s, %bias : f32
      %lg_s = arith.mulf %lg, %scale : f32
      %lg_n = arith.addf %lg_s, %bias : f32
      %lb_s = arith.mulf %lb, %scale : f32
      %lb_n = arith.addf %lb_s, %bias : f32

      // Normalize right: r_ch = R_ch * scale + bias  (6 ops: 3 fmul + 3 fadd)
      %rr_s = arith.mulf %rr, %scale : f32
      %rr_n = arith.addf %rr_s, %bias : f32
      %rg_s = arith.mulf %rg, %scale : f32
      %rg_n = arith.addf %rg_s, %bias : f32
      %rb_s = arith.mulf %rb, %scale : f32
      %rb_n = arith.addf %rb_s, %bias : f32

      // Per-channel differences  (3 ops: 3 fsub)
      %dr = arith.subf %lr_n, %rr_n : f32
      %dg = arith.subf %lg_n, %rg_n : f32
      %db = arith.subf %lb_n, %rb_n : f32

      // Squared differences for SSD cost  (3 ops: 3 fmul)
      %dr2 = arith.mulf %dr, %dr : f32
      %dg2 = arith.mulf %dg, %dg : f32
      %db2 = arith.mulf %db, %db : f32

      // Weighted SSD: cost = w1*dr² + w2*dg² + w3*db²  (5 ops: 3 fmul + 2 fadd)
      %wdr = arith.mulf %dr2, %w1 : f32
      %wdg = arith.mulf %dg2, %w2 : f32
      %wdb = arith.mulf %db2, %w3 : f32
      %sum_rg = arith.addf %wdr, %wdg : f32
      %cost_val = arith.addf %sum_rg, %wdb : f32

      // Gradient feature: horizontal gradient = (lr-lb)*w1 + (rr-rb)*w2
      // (4 ops: 2 fsub + 2 fmul)
      %gl = arith.subf %lr_n, %lb_n : f32
      %gr = arith.subf %rr_n, %rb_n : f32
      %gls = arith.mulf %gl, %w1 : f32
      %grs = arith.mulf %gr, %w2 : f32

      // Combined gradient (1 op: 1 fadd)
      %grad_val = arith.addf %gls, %grs : f32

      // Store results
      affine.store %cost_val, %cost[%i] : memref<64xf32>
      affine.store %grad_val, %grad[%i] : memref<64xf32>
    }

    // Task 1: Simple post-processing — bias addition (light, stays on 1 CGRA)
    affine.for %j = 0 to 64 {
      %a = affine.load %aux_in[%j] : memref<64xf32>
      %b2 = arith.addf %a, %bias : f32
      affine.store %b2, %aux_out[%j] : memref<64xf32>
    }

    return
  }
}

// SERIALIZED:      module {
// SERIALIZED-NEXT:   func.func @stereo_cost_computation(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>, %arg7: memref<64xf32>, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32, %arg12: f32, %arg13: memref<64xf32>, %arg14: memref<64xf32>) {
// SERIALIZED-NEXT:     affine.for %arg15 = 0 to 64 {
// SERIALIZED-NEXT:       %0 = affine.load %arg0[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %1 = affine.load %arg1[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %2 = affine.load %arg2[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %3 = affine.load %arg3[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %4 = affine.load %arg4[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %5 = affine.load %arg5[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %6 = arith.mulf %0, %arg11 : f32
// SERIALIZED-NEXT:       %7 = arith.addf %6, %arg12 : f32
// SERIALIZED-NEXT:       %8 = arith.mulf %1, %arg11 : f32
// SERIALIZED-NEXT:       %9 = arith.addf %8, %arg12 : f32
// SERIALIZED-NEXT:       %10 = arith.mulf %2, %arg11 : f32
// SERIALIZED-NEXT:       %11 = arith.addf %10, %arg12 : f32
// SERIALIZED-NEXT:       %12 = arith.mulf %3, %arg11 : f32
// SERIALIZED-NEXT:       %13 = arith.addf %12, %arg12 : f32
// SERIALIZED-NEXT:       %14 = arith.mulf %4, %arg11 : f32
// SERIALIZED-NEXT:       %15 = arith.addf %14, %arg12 : f32
// SERIALIZED-NEXT:       %16 = arith.mulf %5, %arg11 : f32
// SERIALIZED-NEXT:       %17 = arith.addf %16, %arg12 : f32
// SERIALIZED-NEXT:       %18 = arith.subf %7, %13 : f32
// SERIALIZED-NEXT:       %19 = arith.subf %9, %15 : f32
// SERIALIZED-NEXT:       %20 = arith.subf %11, %17 : f32
// SERIALIZED-NEXT:       %21 = arith.mulf %18, %18 : f32
// SERIALIZED-NEXT:       %22 = arith.mulf %19, %19 : f32
// SERIALIZED-NEXT:       %23 = arith.mulf %20, %20 : f32
// SERIALIZED-NEXT:       %24 = arith.mulf %21, %arg8 : f32
// SERIALIZED-NEXT:       %25 = arith.mulf %22, %arg9 : f32
// SERIALIZED-NEXT:       %26 = arith.mulf %23, %arg10 : f32
// SERIALIZED-NEXT:       %27 = arith.addf %24, %25 : f32
// SERIALIZED-NEXT:       %28 = arith.addf %27, %26 : f32
// SERIALIZED-NEXT:       %29 = arith.subf %7, %11 : f32
// SERIALIZED-NEXT:       %30 = arith.subf %13, %17 : f32
// SERIALIZED-NEXT:       %31 = arith.mulf %29, %arg8 : f32
// SERIALIZED-NEXT:       %32 = arith.mulf %30, %arg9 : f32
// SERIALIZED-NEXT:       %33 = arith.addf %31, %32 : f32
// SERIALIZED-NEXT:       affine.store %28, %arg6[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       affine.store %33, %arg7[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     affine.for %arg15 = 0 to 64 {
// SERIALIZED-NEXT:       %0 = affine.load %arg13[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:       %1 = arith.addf %0, %arg12 : f32
// SERIALIZED-NEXT:       affine.store %1, %arg14[%arg15] : memref<64xf32>
// SERIALIZED-NEXT:     }
// SERIALIZED-NEXT:     return
// SERIALIZED-NEXT:   }
// SERIALIZED-NEXT: }

// TASKFLOW:      module {
// TASKFLOW-NEXT:   func.func @stereo_cost_computation
// TASKFLOW:        %dependency_read_out:6, %dependency_write_out:2 = taskflow.task @Task_0
// TASKFLOW:          affine.for %arg28 = 0 to 64 {
// TASKFLOW:          }
// TASKFLOW:          taskflow.yield
// TASKFLOW:        %dependency_read_out_0, %dependency_write_out_1 = taskflow.task @Task_1
// TASKFLOW:          affine.for %arg18 = 0 to 64 {
// TASKFLOW:          }
// TASKFLOW:          taskflow.yield
// TASKFLOW:        return

// RESOPT:      taskflow.task @Task_0_Task_1_utilfused
// RESOPT-SAME: {cgra_count = 3 : i32, compiled_ii = 1 : i32, steps = 10 : i32, tile_shape = "2x2[(0,0)(1,0)(0,1)]", trip_count = 64 : i32}
// RESOPT:      return

// CGRA Tile Occupation after RESOPT (4x4 grid, col x row):
// +---+---+---+---+
// | 0 | 0 | . | . |   row=0: Task_0_Task_1_utilfused occupies 3 CGRAs
// +---+---+---+---+         in a 2x2 non-rectangular layout:
// | 0 | . | . | . |         (0,0), (1,0), (0,1)
// +---+---+---+---+
// | . | . | . | . |   Total tile array: 8x8 (3 CGRAs × 16 tiles = 48 tiles)
// +---+---+---+---+
// | . | . | . | . |   res_mii=3 (16 tiles) → 2 (32 tiles) → 1 (48 tiles)
// +---+---+---+---+   
// 0=Task_0_Task_1_utilfused (cgra_count=3); 3/16 CGRAs used