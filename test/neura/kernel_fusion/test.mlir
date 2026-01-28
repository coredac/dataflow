// RUN: mlir-neura-opt --wrap-loop-in-kernel --fuse-kernel %s 2>&1 | FileCheck %s

// =============================================================================
// TEST 1: Producer-Consumer Fusion
// Expected: Both loops should be fused into a single kernel.
// =============================================================================

// CHECK-LABEL: func.func @test_producer_consumer_fusion(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK: neura.kernel ins(%arg0, %arg1, %arg2, %cst, %arg3 : memref<?xf32>, memref<?xf32>, memref<?xf32>, f32, memref<?xf32>) attributes {kernel_name = "fused_sibling"} {
// CHECK: affine.for
// CHECK: arith.addf
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK-NOT: neura.kernel
// CHECK: return

func.func @test_producer_consumer_fusion(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
  %cst = arith.constant 2.000000e+00 : f32
  affine.for %arg4 = 0 to 64 {
    %0 = memref.load %arg0[%arg4] : memref<?xf32>
    %1 = memref.load %arg1[%arg4] : memref<?xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %arg2[%arg4] : memref<?xf32>
  }
  affine.for %arg4 = 0 to 64 {
    %0 = memref.load %arg2[%arg4] : memref<?xf32>
    %1 = arith.mulf %0, %cst : f32
    memref.store %1, %arg3[%arg4] : memref<?xf32>
  }
  return
}

// =============================================================================
// TEST 2: Sibling Fusion
// Expected: Both loops should be fused into a single kernel.
// =============================================================================

// CHECK-LABEL: func.func @test_sibling_fusion(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
// CHECK: neura.kernel ins(%arg0, %cst_0, %arg1, %cst, %arg2 : memref<?xf32>, f32, memref<?xf32>, f32, memref<?xf32>) attributes {kernel_name = "fused_sibling"} {
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: affine.for
// CHECK: arith.addf
// CHECK-NOT: neura.kernel
// CHECK: return

func.func @test_sibling_fusion(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 3.000000e+00 : f32
  affine.for %arg3 = 0 to 64 {
    %0 = memref.load %arg0[%arg3] : memref<?xf32>
    %1 = arith.mulf %0, %cst_0 : f32
    memref.store %1, %arg1[%arg3] : memref<?xf32>
  }
  affine.for %arg3 = 0 to 64 {
    %0 = memref.load %arg0[%arg3] : memref<?xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %arg2[%arg3] : memref<?xf32>
  }
  return
}

// =============================================================================
// TEST 3: No Shared Input (No Fusion)
// Expected: Kernels should NOT be fused as siblings since they dont't share input.
// =============================================================================
// CHECK-LABEL: func.func @test_no_shared_input(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK: neura.kernel
// CHECK-SAME: kernel_name = "kernel_0"
// CHECK: neura.kernel
// CHECK-SAME: kernel_name = "kernel_1"
// CHECK: return

func.func @test_no_shared_input(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
  %cst = arith.constant 3.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  affine.for %arg4 = 0 to 64 {
    %0 = memref.load %arg0[%arg4] : memref<?xf32>
    %1 = arith.mulf %0, %cst_0 : f32
    memref.store %1, %arg2[%arg4] : memref<?xf32>
  }
  affine.for %arg4 = 0 to 64 {
    %0 = memref.load %arg1[%arg4] : memref<?xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %arg3[%arg4] : memref<?xf32>
  }
  return
}

// =============================================================================
// TEST 4: Chain fusion: A -> B -> C
// Expected: All kernels should be fused into a single kernel.
// =============================================================================
// CHECK-LABEL: func.func @test_chain_fusion(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
// CHECK: neura.kernel ins(%arg0, %arg1, %arg2, %cst_0, %arg3, %cst, %arg4 : memref<?xf32>, memref<?xf32>, memref<?xf32>, f32, memref<?xf32>, f32, memref<?xf32>) attributes {kernel_name = "fused_sibling"} {
// CHECK: affine.for
// CHECK: arith.addf
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: affine.for
// CHECK: arith.addf
// CHECK-NOT: neura.kernel
// CHECK: return

func.func @test_chain_fusion(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  affine.for %arg5 = 0 to 64 {
    %0 = memref.load %arg0[%arg5] : memref<?xf32>
    %1 = memref.load %arg1[%arg5] : memref<?xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %arg2[%arg5] : memref<?xf32>
  }
  affine.for %arg5 = 0 to 64 {
    %0 = memref.load %arg2[%arg5] : memref<?xf32>
    %1 = arith.mulf %0, %cst_0 : f32
    memref.store %1, %arg3[%arg5] : memref<?xf32>
  }
  affine.for %arg5 = 0 to 64 {
    %0 = memref.load %arg3[%arg5] : memref<?xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %arg4[%arg5] : memref<?xf32>
  }
  return
}

// =============================================================================
// TEST 5: Complex Sibling Fusion
// Expected: Siblings that share inputs should be fused, but kernel_3 should remain as a separate kernel.
// =============================================================================

// CHECK-LABEL: func.func @test_complex_sibling(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>) {
// CHECK: neura.kernel ins(%arg0, %cst_1, %arg2, %cst_0, %arg3, %arg4 : memref<?xf32>, f32, memref<?xf32>, f32, memref<?xf32>, memref<?xf32>) attributes {kernel_name = "fused_sibling"} {
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: affine.for
// CHECK: arith.addf
// CHECK: affine.for
// CHECK: arith.subf
// CHECK: neura.kernel
// CHECK-SAME: kernel_name = "kernel_3"
// CHECK: return

func.func @test_complex_sibling(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>) {
  %cst = arith.constant 3.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg0[%arg6] : memref<?xf32>
    %1 = arith.mulf %0, %cst_1 : f32
    memref.store %1, %arg2[%arg6] : memref<?xf32>
  }
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg0[%arg6] : memref<?xf32>
    %1 = arith.addf %0, %cst_0 : f32
    memref.store %1, %arg3[%arg6] : memref<?xf32>
  }
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg0[%arg6] : memref<?xf32>
    %1 = arith.subf %0, %cst_0 : f32
    memref.store %1, %arg4[%arg6] : memref<?xf32>
  }
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg1[%arg6] : memref<?xf32>
    %1 = arith.mulf %0, %cst : f32
    memref.store %1, %arg5[%arg6] : memref<?xf32>
  }
  return
}

// =============================================================================
// TEST 6: Mixed Patterns
// Expected: All four loops should be fused into a single kernel.
// =============================================================================

// CHECK-LABEL: func.func @test_mixed_patterns(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>) {
// CHECK: neura.kernel ins(%arg0, %arg1, %arg2, %cst_0, %arg3, %cst, %arg4, %arg5 : memref<?xf32>, memref<?xf32>, memref<?xf32>, f32, memref<?xf32>, f32, memref<?xf32>, memref<?xf32>) attributes {kernel_name = "fused_sibling"} {
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// CHECK: affine.for
// CHECK-NOT: neura.kernel
// CHECK: return

func.func @test_mixed_patterns(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>, %arg5: memref<?xf32>) {
  %cst = arith.constant 3.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg0[%arg6] : memref<?xf32>
    %1 = memref.load %arg1[%arg6] : memref<?xf32>
    %2 = arith.addf %0, %1 : f32
    memref.store %2, %arg2[%arg6] : memref<?xf32>
  }
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg0[%arg6] : memref<?xf32>
    %1 = arith.mulf %0, %cst_0 : f32
    memref.store %1, %arg3[%arg6] : memref<?xf32>
  }
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg0[%arg6] : memref<?xf32>
    %1 = arith.addf %0, %cst : f32
    memref.store %1, %arg4[%arg6] : memref<?xf32>
  }
  affine.for %arg6 = 0 to 64 {
    %0 = memref.load %arg2[%arg6] : memref<?xf32>
    %1 = arith.mulf %0, %cst_0 : f32
    memref.store %1, %arg5[%arg6] : memref<?xf32>
  }
  return
}
