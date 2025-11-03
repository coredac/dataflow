// RUN: not mlir-neura-opt %s --lower-affine-to-neura 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: mlir-neura-opt %s --lower-affine | FileCheck %s --check-prefix=CHECK-SCF
// RUN: mlir-neura-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: mlir-neura-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --lower-llvm-to-neura | FileCheck %s --check-prefix=CHECK-NEURA-BR

// Unsupported Case: affine.if conditional
//
// This test demonstrates the complete multi-stage lowering chain for conditionals:
// 1. Direct lowering to Neura (--lower-affine-to-neura) fails with a clear error
// 2. Lowering to SCF (--lower-affine) succeeds, producing scf.if and scf.for
// 3. Further lowering to LLVM succeeds, producing llvm.cond_br and llvm.br
// 4. Lowering LLVM to Neura succeeds, producing neura.cond_br and neura.br
// 5. However, neura.br/neura.cond_br CANNOT be mapped to CGRA hardware
//    because CGRAs lack program counters and branch execution units
//
// The complete transformation chain:
//   affine.if → scf.if → cf.cond_br → llvm.cond_br → neura.cond_br ✓
//   But: neura.br/neura.cond_br → CGRA tiles ❌ (no hardware support)
//
// Neura dialect is designed for spatial dataflow architectures where:
// - Operations are mapped to physical tiles in a 2D array
// - Data flows through interconnect links between tiles  
// - Control flow must use predicated execution (neura.grant_predicate), not branches
module {
  func.func @affine_if_example(%arg0: memref<10xf32>) {
    affine.for %i = 0 to 10 {
      affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
        %val = affine.load %arg0[%i] : memref<10xf32>
      }
    }
    return
  }
}

// ============================================================================
// Test 1: Direct lowering to Neura fails with clear error
// ============================================================================
// CHECK-ERROR: error:
// CHECK-ERROR: affine.if

// ============================================================================
// Test 2: Lowering to SCF succeeds, producing scf.if and scf.for
// ============================================================================
// CHECK-SCF-LABEL: func.func @affine_if_example(%arg0: memref<10xf32>)
// CHECK-SCF-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-SCF-NEXT: %[[C10:.*]] = arith.constant 10 : index
// CHECK-SCF-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-SCF-NEXT: scf.for %[[IV:.*]] = %[[C0]] to %[[C10]] step %[[C1]]
// CHECK-SCF-NEXT:   %[[C0_0:.*]] = arith.constant 0 : index
// CHECK-SCF-NEXT:   %[[C_NEG5:.*]] = arith.constant -5 : index
// CHECK-SCF-NEXT:   %[[ADD:.*]] = arith.addi %[[IV]], %[[C_NEG5]] : index
// CHECK-SCF-NEXT:   %[[CMP:.*]] = arith.cmpi sge, %[[ADD]], %[[C0_0]] : index
// CHECK-SCF-NEXT:   scf.if %[[CMP]]
// CHECK-SCF-NEXT:     %{{.*}} = memref.load %arg0[%[[IV]]] : memref<10xf32>
// CHECK-SCF-NEXT:   }
// CHECK-SCF-NEXT: }
// CHECK-SCF-NEXT: return

// ============================================================================
// Test 3: Lowering to LLVM dialect succeeds, producing llvm.cond_br
// ============================================================================
// CHECK-LLVM-LABEL: llvm.func @affine_if_example
// CHECK-LLVM: llvm.br ^bb1
// CHECK-LLVM: ^bb1
// CHECK-LLVM: llvm.icmp "slt"
// CHECK-LLVM: llvm.cond_br
// CHECK-LLVM: ^bb2
// CHECK-LLVM: llvm.icmp "sge"
// CHECK-LLVM: llvm.cond_br
// CHECK-LLVM: ^bb3
// CHECK-LLVM: llvm.br
// CHECK-LLVM: ^bb4
// CHECK-LLVM: llvm.add
// CHECK-LLVM: llvm.br
// CHECK-LLVM: ^bb5
// CHECK-LLVM: llvm.return

// ============================================================================
// Test 4: Lowering LLVM to Neura succeeds, producing neura.cond_br
// ============================================================================
// CHECK-NEURA-BR-LABEL: llvm.func @affine_if_example
// CHECK-NEURA-BR: neura.br {{.*}} to ^bb1
// CHECK-NEURA-BR: ^bb1
// CHECK-NEURA-BR: neura.icmp
// CHECK-NEURA-BR: neura.cond_br {{.*}} then to ^bb2 else to ^bb5
// CHECK-NEURA-BR: ^bb2
// CHECK-NEURA-BR: neura.add
// CHECK-NEURA-BR: neura.icmp
// CHECK-NEURA-BR: neura.cond_br {{.*}} then to ^bb3 else to ^bb4
// CHECK-NEURA-BR: ^bb3
// CHECK-NEURA-BR: neura.br to ^bb4
// CHECK-NEURA-BR: ^bb4
// CHECK-NEURA-BR: neura.add
// CHECK-NEURA-BR: neura.br {{.*}} to ^bb1
// CHECK-NEURA-BR: ^bb5
// CHECK-NEURA-BR: neura.return

//
// ============================================================================
// Why neura.br/neura.cond_br cannot map to CGRA hardware
// ============================================================================
// The complete lowering chain successfully transforms through all IR levels:
//   Step 1: affine.if → scf.if (structured control flow)
//   Step 2: scf.if → cf.cond_br (unstructured control flow graph)
//   Step 3: cf.cond_br → llvm.cond_br (LLVM IR level)
//   Step 4: llvm.cond_br → neura.cond_br (Neura dialect level)  ✓
//   Step 5: neura.br/neura.cond_br → CGRA tiles  ❌ (NO hardware mapping)
//
// While neura.br and neura.cond_br exist in the Neura dialect, they CANNOT
// be mapped to physical CGRA hardware because:
// - CGRA tiles are spatial compute units without program counters
// - There are no branch execution units or instruction sequencing logic
// - The dataflow model requires all operations to be spatially placed
// - Dynamic control flow requires runtime decisions incompatible with static routing
//
// These branch operations remain as intermediate representations that:
// 1. Cannot pass the --map-to-accelerator pass (mapping will fail)
// 2. Cannot be converted to CGRA assembly/configuration
// 3. Exist only for completeness of the dialect's IR representation
//
// Future work to support conditionals requires fundamentally different approaches:
// - If-conversion: Transform control flow into data flow with select operations
// - Loop unrolling: Eliminate dynamic branches through compile-time expansion
// - Predicated execution: Use neura.grant_predicate for conditional operations
// - Hybrid execution: Handle control flow on host CPU, dataflow on CGRA

