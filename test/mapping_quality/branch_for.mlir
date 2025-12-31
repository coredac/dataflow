// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   | FileCheck %s

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-live-in \
// RUN:   | FileCheck %s -check-prefix=CANONICALIZE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   | FileCheck %s -check-prefix=CTRL2DATA

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   | FileCheck %s -check-prefix=FUSE

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   | FileCheck %s -check-prefix=MOV

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   --architecture-spec=%S/../arch_spec/architecture.yaml \
// RUN:   | FileCheck %s -check-prefix=MAPPING

// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --promote-func-arg-to-const \
// RUN:   --fold-constant \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --fold-constant \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic backtrack-config=simple" \
// RUN:   --architecture-spec=%S/../arch_spec/architecture.yaml \
// RUN:   --generate-code
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml -check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM

func.func @loop_test() -> f32 {
  %n = llvm.mlir.constant(10 : i64) : i64
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c1 = llvm.mlir.constant(1 : i64) : i64
  %c1f = llvm.mlir.constant(3.0 : f32) : f32
  %acc_init = llvm.mlir.constant(0.0 : f32) : f32

  llvm.br ^bb1(%c0, %acc_init : i64, f32)

^bb1(%i: i64, %acc: f32):  // loop body + check + increment
  %next_acc = llvm.fadd %acc, %c1f : f32
  %i_next = llvm.add %i, %c1 : i64
  %cmp = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %next_acc : i64, f32), ^exit(%next_acc : f32)

^exit(%result: f32):
  return %result : f32
}

// CHECK:      func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CHECK-NEXT:   %0 = "neura.constant"() <{value = 10 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %1 = "neura.constant"() <{value = 0 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %2 = "neura.constant"() <{value = 1 : i64}> : () -> !neura.data<i64, i1>
// CHECK-NEXT:   %3 = "neura.constant"() <{value = 3.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   %4 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> !neura.data<f32, i1>
// CHECK-NEXT:   neura.br %1, %4 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1
// CHECK-NEXT: ^bb1(%5: !neura.data<i64, i1>, %6: !neura.data<f32, i1>):
// CHECK-NEXT:   %7 = "neura.fadd"(%6, %3) : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// CHECK-NEXT:   %8 = "neura.add"(%5, %2) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// CHECK-NEXT:   %9 = "neura.icmp"(%8, %0) <{cmpType = "slt"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// CHECK-NEXT:   neura.cond_br %9 : !neura.data<i1, i1> then %8, %7 : !neura.data<i64, i1>, !neura.data<f32, i1> to ^bb1 else %7 : !neura.data<f32, i1> to ^bb2
// CHECK-NEXT: ^bb2(%10: !neura.data<f32, i1>):
// CHECK-NEXT:   "neura.return"(%10) : (!neura.data<f32, i1>) -> ()
// CHECK-NEXT: }

// CANONICALIZE:       func.func @loop_test() -> f32 attributes {accelerator = "neura"} {
// CANONICALIZE-NEXT:     %0 = "neura.constant"() <{value = 0 : i64}> : () -> i64
// CANONICALIZE-NEXT:     %1 = "neura.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
// CANONICALIZE-NEXT:     neura.br %0, %1 : i64, f32 to ^bb1
// CANONICALIZE-NEXT:   ^bb1(%2: i64, %3: f32):
// CANONICALIZE-NEXT:     %4 = "neura.fadd"(%3) {rhs_value = 3.000000e+00 : f32} : (f32) -> f32
// CANONICALIZE-NEXT:     %5 = "neura.add"(%2) {rhs_value = 1 : i64} : (i64) -> i64
// CANONICALIZE-NEXT:     %6 = "neura.icmp"(%5) <{cmpType = "slt"}> {rhs_value = 10 : i64} : (i64) -> i1
// CANONICALIZE-NEXT:     neura.cond_br %6 : i1 then %5, %4 : i64, f32 to ^bb1 else %4 : f32 to ^bb2
// CANONICALIZE-NEXT:   ^bb2(%7: f32):
// CANONICALIZE-NEXT:     "neura.return"(%7) : (f32) -> ()
// CANONICALIZE-NEXT:   }

// CTRL2DATA:        func.func @loop_test()
// CTRL2DATA-SAME:   accelerator = "neura"
// CTRL2DATA-SAME:   dataflow_mode = "predicate"
// CTRL2DATA:        neura.return_value

// FUSE:        func.func @loop_test()
// FUSE-SAME:   accelerator = "neura"
// FUSE-SAME:   dataflow_mode = "predicate"

// MOV:        func.func @loop_test()
// MOV-SAME:   accelerator = "neura"
// MOV-SAME:   dataflow_mode = "predicate"
// MOV:        %[[PHI_ACC:.+]] = neura.phi_start
// MOV:        %[[PHI_I:.+]] = neura.phi_start
// MOV:        %[[FADD:.+]] = "neura.fadd"
// MOV:        %[[ADD:.+]] = "neura.add"
// MOV:        %[[ICMP:.+]] = "neura.icmp"
// MOV:        neura.return_value

// MAPPING: func.func @loop_test()
// MAPPING-SAME: accelerator = "neura"
// MAPPING-SAME: mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}

// YAML:      array_config:
// YAML-NEXT:     columns: 4
// YAML-NEXT:     rows: 4
// YAML-NEXT:     compiled_ii: 4

// ASM:      # Compiled II: 4
// ASM:     PE(0,0):
// ASM-NEXT:     {
// ASM-NEXT:     GRANT_ONCE, [#0] -> [EAST, RED] (t=0, inv_iters=0)
// ASM-NEXT:     DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=4, inv_iters=1)
// ASM-NEXT:     } (idx_per_ii=0)
