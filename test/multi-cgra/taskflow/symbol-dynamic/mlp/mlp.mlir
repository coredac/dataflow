// RUN: python3 %S/mlp_pipeline.py %t.linalg.mlir

// RUN: neura-compiler %t.linalg.mlir \
// RUN:   --linalg-to-affine-conversion \
// RUN:   -o %t.affine.mlir
// RUN: FileCheck --input-file=%t.affine.mlir %s --check-prefix=AFFINE

// AFFINE:          %dim_3 = memref.dim %arg0, %c0 : memref<?x64xf32>
// AFFINE-NEXT:     affine.for %arg1 = 0 to %dim_3 {
// AFFINE-NEXT:       affine.for %arg2 = 0 to 128 {
// AFFINE-NEXT:         affine.for %arg3 = 0 to 64 {
// AFFINE-NEXT:           %3 = affine.load %arg0[%arg1, %arg3] : memref<?x64xf32>
// AFFINE-NEXT:           %4 = affine.load %alloc[%arg3, %arg2] : memref<64x128xf32>
// AFFINE-NEXT:           %5 = affine.load %alloc_2[%arg1, %arg2] : memref<?x128xf32>
// AFFINE-NEXT:           %6 = arith.mulf %3, %4 : f32
// AFFINE-NEXT:           %7 = arith.addf %5, %6 : f32
// AFFINE-NEXT:           affine.store %7, %alloc_2[%arg1, %arg2] : memref<?x128xf32>
// AFFINE-NEXT:         }
// AFFINE-NEXT:       }
// AFFINE-NEXT:     }