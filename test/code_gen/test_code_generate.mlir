// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-return \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=../arch_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir
// RUN: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
// RUN: FileCheck %s --input-file=tmp-generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=tmp-generated-instructions.asm --check-prefix=ASM


func.func @loop_test() -> f32 {
  %n        = llvm.mlir.constant(10 : i64) : i64
  %c0       = llvm.mlir.constant(0  : i64) : i64
  %c1       = llvm.mlir.constant(1  : i64) : i64
  %c1f      = llvm.mlir.constant(3.0 : f32) : f32
  %acc_init = llvm.mlir.constant(0.0 : f32) : f32

  llvm.br ^bb1(%c0, %acc_init : i64, f32)

^bb1(%i: i64, %acc: f32):
  %next_acc = llvm.fadd %acc, %c1f : f32
  %i_next   = llvm.add  %i, %c1    : i64
  %cmp      = llvm.icmp "slt" %i_next, %n : i64
  llvm.cond_br %cmp, ^bb1(%i_next, %next_acc : i64, f32), ^exit(%next_acc : f32)

^exit(%result: f32):
  return %result : f32
}

// MAPPING: func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {

// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   compiled_ii: 4
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - index_per_ii: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   id: 1
// YAML-NEXT:                   time_step: 0
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   id: 17
// YAML-NEXT:                   time_step: 1
// YAML-NEXT:                   invalid_iterations: 0
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   id: 36
// YAML-NEXT:                   time_step: 5
// YAML-NEXT:                   invalid_iterations: 1
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - index_per_ii: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   id: 55
// YAML-NEXT:                   time_step: 7

// ASM: # Compiled II: 4
// ASM: PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [EAST, RED] (t=1, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [$0], [NORTH, RED] -> [NORTH, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM: PE(1,0):
// ASM-NEXT: {
// ASM-NEXT:   DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=4, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [WEST, RED], [EAST, RED] -> [NORTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM: PE(2,0):
// ASM-NEXT: {
// ASM-NEXT:   GRANT_PREDICATE, [WEST, RED], [NORTH, RED] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [EAST, RED], [NORTH, RED] -> [NORTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM: PE(3,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#10] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   GRANT_ONCE, [$0] -> [WEST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM: PE(0,1):
// ASM-NEXT: {
// ASM-NEXT:   PHI_START, [$0], [SOUTH, RED] -> [$0], [SOUTH, RED] (t=4, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT:   FADD, [NORTH, RED], [$0] -> [NORTH, RED], [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#3.000000] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT:   DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)

