// RUN: mlir-neura-opt %s --convert-affine-to-taskflow \
// RUN: --construct-hyperblock-from-task \
// RUN: --canonicalize-task \
// RUN: --analyze-mct-dependency 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPENDENCY

// Test for MCT dependency analysis pass.
// This test verifies that the pass correctly identifies:
// 1. SSA dependencies between tasks
// 2. Same-header fusion candidates

module {
  func.func @dependency_test(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    
    // First loop: writes to arg0
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 8 {
        %idx = arith.index_cast %i : index to i32
        affine.store %idx, %arg0[%i, %j] : memref<4x8xi32>
      }
    }

    // Second loop: reads from arg0, writes to arg1 (same header as third loop)
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 8 {
        %loaded = affine.load %arg0[%i, %j] : memref<4x8xi32>
        affine.store %loaded, %arg1[%i, %j] : memref<4x8xi32>
      }
    }

    return
  }
}

// DEPENDENCY: === MCT Dependency Analysis ===
// DEPENDENCY: Found 2 MCTs
// DEPENDENCY: MCT 0: Task_0
// DEPENDENCY: MCT 1: Task_1
// DEPENDENCY: === Dependencies ===
// DEPENDENCY: Task_0 → Task_1 : SSA
// DEPENDENCY: === Summary ===
