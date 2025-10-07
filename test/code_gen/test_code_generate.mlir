// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --architecture-spec=../test_architecture_spec/architecture.yaml \
// RUN:   --generate-code -o %t-mapping.mlir 
// RU: FileCheck %s --input-file=%t-mapping.mlir -check-prefix=MAPPING
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

// MAPPING:        func.func @loop_test() -> f32 attributes {accelerator = "neura", mapping_info = {compiled_ii = 6 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {

// Each core represents a processing element in the CGRA array
// Example: column: 1, row: 1 represents the core at position (1,1) in the 4x4 grid
// Each core contains multiple entries (execution contexts) with instructions
// Instructions are organized by timestep and include source/destination operands
// Tile (1,1) : per-cycle schedule and routing summary.
//
// entry0 @ t=2:
//   PHI merges tokens arriving from EAST and SOUTH, then forwards the selected
//   value out to EAST.
//
// entry1 @ t=4:
//   ICMP consumes EAST and SOUTH, then BROADCASTS its result:
//     - to EAST / NORTH / SOUTH (for downstream tiles),
//     - and into local registers $22, $21, $20 (to retain the value for later use).
//
// entry2 @ t=5:
//   NOT reads temporary $20 and forwards the negated value to EAST.
//
// entry3 @ t=6:
//   GRANT_PREDICATE uses a control input from WEST together with predicate
//   state latched in $21, and forwards the grant out to EAST.
//
// entry4 @ t=6:
//   DATA_MOV performs a register deposit: value arriving from SOUTH is written
//   into local register $20.
//
// entry5 @ t=7:
//   GRANT_PREDICATE combines the recently updated $20 with $22 to produce a
//   new grant and forwards it to EAST.
//
// entry6 @ t=7:
//   CTRL_MOV performs a control deposit: control token arriving from WEST is
//   written into local register $20.
//

// YAML:          - column: 1
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "1"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$32"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$33"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$32"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$32"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$32"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$33"
// YAML-NEXT:                       color: "RED"


// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#0] -> [EAST, RED]
// ASM-NEXT: } (t=0)
// ASM-NEXT: {
// ASM-NEXT:   CONSTANT, [#10] -> [EAST, RED]
// ASM-NEXT: } (t=1)