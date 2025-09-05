// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --generate-code
// RUN: FileCheck %s --input-file=generated-instructions.yaml --check-prefix=YAML
// RUN: FileCheck %s --input-file=generated-instructions.asm --check-prefix=ASM


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


// YAML: array_config:
// YAML-NEXT:   columns: 4
// YAML-NEXT:   rows: 4
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "0"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CONSTANT"
// YAML-NEXT:               timestep: 0
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "#0"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CONSTANT"
// YAML-NEXT:               timestep: 1
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "#10"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 4
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "1"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_ONCE"
// YAML-NEXT:               timestep: 1
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_ONCE"
// YAML-NEXT:               timestep: 2
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$4"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "PHI"
// YAML-NEXT:               timestep: 3
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$5"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$4"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$4"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry3"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 3
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry4"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_PREDICATE"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$4"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$5"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "2"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CONSTANT"
// YAML-NEXT:               timestep: 0
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "#1"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$8"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_ONCE"
// YAML-NEXT:               timestep: 1
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$8"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$8"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "PHI"
// YAML-NEXT:               timestep: 2
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$8"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 0
// YAML-NEXT:       core_id: "3"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "RETURN"
// YAML-NEXT:               timestep: 8
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "4"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "5"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "PHI"
// YAML-NEXT:               timestep: 2
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "ICMP"
// YAML-NEXT:               timestep: 4
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$20"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$21"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$22"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "NOT"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$20"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry3"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_PREDICATE"
// YAML-NEXT:               timestep: 6
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$21"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry4"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 6
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$20"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry5"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_PREDICATE"
// YAML-NEXT:               timestep: 7
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$20"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$22"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry6"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CTRL_MOV"
// YAML-NEXT:               timestep: 7
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$20"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "ADD"
// YAML-NEXT:               timestep: 3
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$25"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "PHI"
// YAML-NEXT:               timestep: 4
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$24"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$24"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "FADD"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$24"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$26"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry3"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$24"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry4"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_PREDICATE"
// YAML-NEXT:               timestep: 6
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$25"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$24"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry5"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 6
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry6"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_PREDICATE"
// YAML-NEXT:               timestep: 7
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$26"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$24"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry7"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 7
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry8"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 8
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "7"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_ONCE"
// YAML-NEXT:               timestep: 3
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 6
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$28"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_PREDICATE"
// YAML-NEXT:               timestep: 7
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "$28"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "9"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 5
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "EAST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "10"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "GRANT_ONCE"
// YAML-NEXT:               timestep: 3
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "NORTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$40"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry1"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "PHI"
// YAML-NEXT:               timestep: 4
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "$40"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry2"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "DATA_MOV"
// YAML-NEXT:               timestep: 6
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "WEST"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:         - entry_id: "entry3"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CTRL_MOV"
// YAML-NEXT:               timestep: 9
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "$41"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 3
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "11"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CONSTANT"
// YAML-NEXT:               timestep: 2
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "#0.000000"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "14"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - opcode: "CONSTANT"
// YAML-NEXT:               timestep: 2
// YAML-NEXT:               src_operands:
// YAML-NEXT:                 - operand: "#3.000000"
// YAML-NEXT:                   color: "RED"
// YAML-NEXT:               dst_operands:
// YAML-NEXT:                 - operand: "SOUTH"
// YAML-NEXT:                   color: "RED"

// ASM-LABEL: PE(0,0):
// ASM: CONSTANT, [#0] -> [EAST] (t=0)
// ASM: CONSTANT, [#10] -> [EAST] (t=1)
// ASM: DATA_MOV, [EAST] -> [NORTH] (t=4)

// ASM-LABEL: PE(1,0):
// ASM: GRANT_ONCE, [WEST] -> [NORTH] (t=1)
// ASM: GRANT_ONCE, [WEST] -> [$4] (t=2)
// ASM: PHI, [$5], [$4] -> [$4], [NORTH] (t=3)
// ASM: DATA_MOV, [EAST] -> [WEST] (t=3)
// ASM: GRANT_PREDICATE, [$4], [NORTH] -> [$5] (t=5)

// ASM-LABEL: PE(2,0):
// ASM: CONSTANT, [#1] -> [$8] (t=0)
// ASM: GRANT_ONCE, [$8] -> [$8] (t=1)
// ASM: PHI, [NORTH], [$8] -> [WEST], [NORTH] (t=2)

// ASM-LABEL: PE(3,0):
// ASM: RETURN, [NORTH] (t=8)

// ASM-LABEL: PE(0,1):
// ASM: DATA_MOV, [SOUTH] -> [EAST] (t=5)

// ASM-LABEL: PE(1,1):
// ASM: PHI, [EAST], [SOUTH] -> [EAST] (t=2)
// ASM: ICMP, [EAST], [SOUTH] -> [$20], [SOUTH], [$21], [$22], [NORTH], [EAST] (t=4)
// ASM: NOT, [$20] -> [EAST] (t=5)
// ASM: GRANT_PREDICATE, [WEST], [$21] -> [EAST] (t=6)
// ASM: DATA_MOV, [NORTH] -> [$20] (t=6)
// ASM: GRANT_PREDICATE, [$20], [$22] -> [EAST] (t=7)
// ASM: CTRL_MOV, [EAST] -> [$20] (t=7)

// ASM-LABEL: PE(2,1):
// ASM: ADD, [WEST], [SOUTH] -> [$25], [WEST] (t=3)
// ASM: PHI, [$24], [EAST] -> [$24] (t=4)
// ASM: FADD, [$24], [NORTH] -> [EAST], [$26] (t=5)
// ASM: DATA_MOV, [WEST] -> [$24] (t=5)
// ASM: GRANT_PREDICATE, [$25], [$24] -> [WEST] (t=6)
// ASM: DATA_MOV, [WEST] -> [EAST] (t=6)
// ASM: GRANT_PREDICATE, [$26], [NORTH] -> [$24] (t=7)
// ASM: DATA_MOV, [WEST] -> [SOUTH] (t=7)
// ASM: DATA_MOV, [WEST] -> [NORTH] (t=8)

// ASM-LABEL: PE(3,1):
// ASM: GRANT_ONCE, [NORTH] -> [WEST] (t=3)
// ASM: DATA_MOV, [WEST] -> [$28] (t=6)
// ASM: GRANT_PREDICATE, [$28], [WEST] -> [SOUTH] (t=7)

// ASM-LABEL: PE(1,2):
// ASM: DATA_MOV, [SOUTH] -> [EAST] (t=5)
// ASM: DATA_MOV, [EAST] -> [SOUTH] (t=5)

// ASM-LABEL: PE(2,2):
// ASM: GRANT_ONCE, [NORTH] -> [$40] (t=3)
// ASM: PHI, [SOUTH], [$40] -> [WEST], [SOUTH] (t=4)
// ASM: DATA_MOV, [WEST] -> [SOUTH] (t=6)
// ASM: CTRL_MOV, [SOUTH] -> [$41] (t=9)

// ASM-LABEL: PE(3,2):
// ASM: CONSTANT, [#0.000000] -> [SOUTH] (t=2)

// ASM-LABEL: PE(2,3):
// ASM: CONSTANT, [#3.000000] -> [SOUTH] (t=2)


