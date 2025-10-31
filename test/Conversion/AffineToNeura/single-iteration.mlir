// RUN: mlir-neura-opt %s --lower-affine-to-neura | FileCheck %s

// Corner Case: Single iteration loop
module {
  func.func @single_iteration(%arg0: memref<1xf32>) {
    affine.for %i = 0 to 1 {
      %0 = affine.load %arg0[%i] : memref<1xf32>
    }
    return
  }
}

// ============================================================================
// Expected output after --lower-affine-to-neura transformation:
// Verify: 1) no affine ops, 2) all neura ops present, 3) exact IR match
// ============================================================================
// CHECK-LABEL: func.func @single_iteration(%arg0: memref<1xf32>)
// CHECK-NEXT: %[[CONST:.*]] = "neura.constant"() <{value = true}> : () -> i1
// CHECK-NEXT: %[[NEXT:.*]], %[[VALID:.*]] = "neura.loop_control"(%[[CONST]]) <{end = 1 : i64, iterationType = "increment", start = 0 : i64, step = 1 : i64}> : (i1) -> (index, i1)
// CHECK-NEXT: %{{.*}} = neura.load_indexed %arg0[%[[NEXT]] : index] memref<1xf32> : f32
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NOT: affine.
