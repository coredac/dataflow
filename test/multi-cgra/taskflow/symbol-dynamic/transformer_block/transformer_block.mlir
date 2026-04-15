// RUN: python3 %S/transformer_block.py %t.linalg.mlir

// RUN: neura-compiler %t.linalg.mlir \
// RUN:   --linalg-to-affine-conversion \
// RUN:   -o %t.affine.mlir
// RUN: FileCheck --input-file=%t.affine.mlir %s --check-prefix=AFFINE

// AFFINE:          %dim_5 = memref.dim %arg0, %c0 : memref<?x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to %dim_5 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 64 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 64 {
// AFFINE-NEXT:           %6 = affine.load %arg0[%arg1, %arg3] : memref<?x64xf32>
// AFFINE-NEXT:           %7 = affine.load %alloc[%arg3, %arg2] : memref<64x64xf32>
// AFFINE-NEXT:           %8 = affine.load %alloc_4[%arg1, %arg2] : memref<?x64xf32>
// AFFINE-NEXT:           %9 = arith.mulf %6, %7 : f32
// AFFINE-NEXT:           %10 = arith.addf %8, %9 : f32
// AFFINE-NEXT:           affine.store %10, %alloc_4[%arg1, %arg2] : memref<?x64xf32>
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }