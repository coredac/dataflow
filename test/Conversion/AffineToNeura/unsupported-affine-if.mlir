// RUN: mlir-neura-opt %s --lower-affine | FileCheck %s --check-prefix=CHECK-SCF
// RUN: mlir-neura-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: mlir-neura-opt %s --lower-affine --convert-scf-to-cf --convert-cf-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --lower-llvm-to-neura | FileCheck %s --check-prefix=CHECK-NEURA-BR

// This test demonstrates the complete multi-stage lowering chain for conditionals.
// Note: Direct lowering affine.if to Neura is not supported.
// 
// The complete transformation chain:
//   affine.if → scf.if → cf.cond_br → llvm.cond_br → neura.cond_br
//
// While neura.cond_br operations are generated, they cannot be mapped to CGRA
// hardware because CGRAs are spatial dataflow architectures without program
// counters or branch prediction units.

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

// CHECK-SCF-LABEL: func.func @affine_if_example(%arg0: memref<10xf32>)
// CHECK-SCF-NEXT: %c0 = arith.constant 0 : index
// CHECK-SCF-NEXT: %c10 = arith.constant 10 : index
// CHECK-SCF-NEXT: %c1 = arith.constant 1 : index
// CHECK-SCF-NEXT: scf.for %arg1 = %c0 to %c10 step %c1
// CHECK-SCF-NEXT:   %c0_0 = arith.constant 0 : index
// CHECK-SCF-NEXT:   %c-5 = arith.constant -5 : index
// CHECK-SCF-NEXT:   %0 = arith.addi %arg1, %c-5 : index
// CHECK-SCF-NEXT:   %1 = arith.cmpi sge, %0, %c0_0 : index
// CHECK-SCF-NEXT:   scf.if %1
// CHECK-SCF-NEXT:     %2 = memref.load %arg0[%arg1] : memref<10xf32>
// CHECK-SCF-NEXT:   }
// CHECK-SCF-NEXT: }
// CHECK-SCF-NEXT: return

// CHECK-LLVM-LABEL: llvm.func @affine_if_example
// CHECK-LLVM: %{{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-LLVM: %{{.*}} = llvm.mlir.constant(10 : index) : i64
// CHECK-LLVM: %{{.*}} = llvm.mlir.constant(1 : index) : i64
// CHECK-LLVM: llvm.br ^bb1(%{{.*}} : i64)
// CHECK-LLVM: ^bb1(%{{.*}}: i64):
// CHECK-LLVM: %{{.*}} = llvm.icmp "slt" %{{.*}}, %{{.*}} : i64
// CHECK-LLVM: llvm.cond_br %{{.*}}, ^bb2, ^bb5
//
// CHECK-LLVM: ^bb2:
// CHECK-LLVM: %{{.*}} = llvm.mlir.constant(0 : index) : i64
// CHECK-LLVM: %{{.*}} = llvm.mlir.constant(-5 : index) : i64
// CHECK-LLVM: %{{.*}} = llvm.add %{{.*}}, %{{.*}} : i64
// CHECK-LLVM: %{{.*}} = llvm.icmp "sge" %{{.*}}, %{{.*}} : i64
// CHECK-LLVM: llvm.cond_br %{{.*}}, ^bb3, ^bb4

// CHECK-NEURA-BR-LABEL: llvm.func @affine_if_example
// CHECK-NEURA-BR: %{{.*}} = "neura.constant"() <{value = -5 : index}> : () -> i64
// CHECK-NEURA-BR: %{{.*}} = "neura.constant"() <{value = 1 : index}> : () -> i64
// CHECK-NEURA-BR: %{{.*}} = "neura.constant"() <{value = 10 : index}> : () -> i64
// CHECK-NEURA-BR: %{{.*}} = "neura.constant"() <{value = 0 : index}> : () -> i64
// CHECK-NEURA-BR: neura.br %{{.*}} : i64 to ^bb1
// CHECK-NEURA-BR: ^bb1(%{{.*}}: i64):
// CHECK-NEURA-BR: %{{.*}} = "neura.icmp"(%{{.*}}, %{{.*}}) <{cmpType = "slt"}> : (i64, i64) -> i1
// CHECK-NEURA-BR: neura.cond_br %{{.*}} : i1 then to ^bb2 else to ^bb5
//
// CHECK-NEURA-BR: ^bb2:
// CHECK-NEURA-BR: %{{.*}} = "neura.add"(%{{.*}}, %{{.*}}) : (i64, i64) -> i64
// CHECK-NEURA-BR: %{{.*}} = "neura.icmp"(%{{.*}}, %{{.*}}) <{cmpType = "sge"}> : (i64, i64) -> i1
// CHECK-NEURA-BR: neura.cond_br %{{.*}} : i1 then to ^bb3 else to ^bb4
