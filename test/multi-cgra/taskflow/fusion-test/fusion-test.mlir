// Test file for fuse-task pass with different fusion scenarios.
//
// Pipeline to produce tasks:
// RUN: mlir-neura-opt %s \
// RUN:   --affine-loop-tree-serialization \
// RUN:   --convert-affine-to-taskflow \
// RUN:   --construct-hyperblock-from-task \
// RUN:   -o %t.taskflow.mlir
//
// Pipeline to test fusion:
// RUN: mlir-neura-opt %s \
// RUN:   --affine-loop-tree-serialization \
// RUN:   --convert-affine-to-taskflow \
// RUN:   --construct-hyperblock-from-task \
// RUN:   --fuse-task \
// RUN:   --architecture-spec=%S/../../../arch_spec/architecture.yaml \
// RUN:   -o %t.fused.mlir

module attributes {} {

// Case 1: Producer-Consumer (fusible)
//   Loop 1 writes to B; Loop 2 reads from B and writes to C.
//   Same loop bounds (0..4), so loops can be merged.
func.func @producer_consumer(
    %A: memref<4xi32>, %B: memref<4xi32>, %C: memref<4xi32>) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  affine.for %i = 0 to 4 {
    %v = affine.load %A[%i] : memref<4xi32>
    %r = arith.addi %v, %c1 : i32
    affine.store %r, %B[%i] : memref<4xi32>
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %B[%i] : memref<4xi32>
    %r = arith.muli %v, %c2 : i32
    affine.store %r, %C[%i] : memref<4xi32>
  }
  %c0 = arith.constant 0 : index
  %ret = memref.load %C[%c0] : memref<4xi32>
  return %ret : i32
}

// Case 2: Sibling tasks (fusible)
//   Both loops read from A, write to different outputs. Same bounds.
func.func @sibling_fusion(
    %A: memref<4xi32>, %B: memref<4xi32>, %C: memref<4xi32>) -> i32 {
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  affine.for %i = 0 to 4 {
    %v = affine.load %A[%i] : memref<4xi32>
    %r = arith.addi %v, %c1 : i32
    affine.store %r, %B[%i] : memref<4xi32>
  }
  affine.for %i = 0 to 4 {
    %v = affine.load %A[%i] : memref<4xi32>
    %r = arith.subi %v, %c3 : i32
    affine.store %r, %C[%i] : memref<4xi32>
  }
  %c0 = arith.constant 0 : index
  %ret = memref.load %B[%c0] : memref<4xi32>
  return %ret : i32
}

// Case 3: Different loop bounds (not fusible)
//   Loops have different trip counts: 0..4 vs 0..8.
func.func @different_bounds(
    %A: memref<4xi32>, %B: memref<4xi32>,
    %C: memref<8xi32>, %D: memref<8xi32>) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  affine.for %i = 0 to 4 {
    %v = affine.load %A[%i] : memref<4xi32>
    %r = arith.addi %v, %c1 : i32
    affine.store %r, %B[%i] : memref<4xi32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %C[%i] : memref<8xi32>
    %r = arith.muli %v, %c2 : i32
    affine.store %r, %D[%i] : memref<8xi32>
  }
  %c0 = arith.constant 0 : index
  %ret = memref.load %B[%c0] : memref<4xi32>
  return %ret : i32
}

}
