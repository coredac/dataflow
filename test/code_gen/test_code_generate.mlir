// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
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

// MAPPING:        func.func @loop_test() -> f32 attributes {accelerator = "neura", dataflow_mode = "predicate", mapping_info = {compiled_ii = 4 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 2 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}} {

// Each core represents a processing element in the CGRA array
// Example: column: 1, row: 1 represents the core at position (1,1) in the 4x4 grid
// Each core contains multiple entries (execution contexts) with instructions
// Instructions are organized by timestep and include source/destination operands
// Tile (1,1) : per-cycle schedule and routing summary.
//
// entry0 @ t=2:
//   PHI_START merges tokens arriving from EAST and SOUTH, then forwards the selected
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

// YAML: array_config:
// YAML:   columns: 4
// YAML:   rows: 4
// YAML:   compiled_ii: 4
// YAML:   cores:
// YAML:     - column: 0
// YAML:       row: 0
// YAML:       core_id: "0"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "CONSTANT"
// YAML:                   id: 1
// YAML:                   time_step: 0
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "#0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "GRANT_ONCE"
// YAML:                   id: 16
// YAML:                   time_step: 1
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 35
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "GRANT_PREDICATE"
// YAML:                   id: 54
// YAML:                   time_step: 7
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:     - column: 1
// YAML:       row: 0
// YAML:       core_id: "1"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 400001
// YAML:                   time_step: 4
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "PHI_START"
// YAML:                   id: 26
// YAML:                   time_step: 2
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:     - column: 2
// YAML:       row: 0
// YAML:       core_id: "2"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "GRANT_PREDICATE"
// YAML:                   id: 56
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "PHI_START"
// YAML:                   id: 25
// YAML:                   time_step: 3
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:     - column: 3
// YAML:       row: 0
// YAML:       core_id: "3"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "CONSTANT"
// YAML:                   id: 0
// YAML:                   time_step: 1
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "#10"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "GRANT_ONCE"
// YAML:                   id: 15
// YAML:                   time_step: 2
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:     - column: 0
// YAML:       row: 1
// YAML:       core_id: "4"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "PHI_START"
// YAML:                   id: 28
// YAML:                   time_step: 4
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "FADD"
// YAML:                   id: 39
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "CONSTANT"
// YAML:                   id: 3
// YAML:                   time_step: 2
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "#3.000000"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 480002
// YAML:                   time_step: 6
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "GRANT_ONCE"
// YAML:                   id: 18
// YAML:                   time_step: 3
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:     - column: 1
// YAML:       row: 1
// YAML:       core_id: "5"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "CONSTANT"
// YAML:                   id: 2
// YAML:                   time_step: 0
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "#1"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "GRANT_ONCE"
// YAML:                   id: 17
// YAML:                   time_step: 1
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 480001
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "PHI_START"
// YAML:                   id: 27
// YAML:                   time_step: 2
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 420001
// YAML:                   time_step: 6
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "ADD"
// YAML:                   id: 38
// YAML:                   time_step: 3
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 570002
// YAML:                   time_step: 7
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:     - column: 2
// YAML:       row: 1
// YAML:       core_id: "6"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "ICMP_SLT"
// YAML:                   id: 44
// YAML:                   time_step: 4
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                     - operand: "$2"
// YAML:                       color: "RED"
// YAML:                     - operand: "$1"
// YAML:                       color: "RED"
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "GRANT_PREDICATE"
// YAML:                   id: 53
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "$2"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "GRANT_PREDICATE"
// YAML:                   id: 52
// YAML:                   time_step: 6
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "$1"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 570001
// YAML:                   time_step: 6
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 33
// YAML:                   time_step: 3
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:     - column: 3
// YAML:       row: 1
// YAML:       core_id: "7"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "NOT"
// YAML:                   id: 51
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:     - column: 0
// YAML:       row: 2
// YAML:       core_id: "8"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "PHI_START"
// YAML:                   id: 29
// YAML:                   time_step: 4
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "NORTH"
// YAML:                       color: "RED"
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 430001
// YAML:                   time_step: 6
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:     - column: 1
// YAML:       row: 2
// YAML:       core_id: "9"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 0
// YAML:               operations:
// YAML:                 - opcode: "GRANT_PREDICATE"
// YAML:                   id: 63
// YAML:                   time_step: 8
// YAML:                   invalid_iterations: 2
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "RETURN"
// YAML:                   id: 65
// YAML:                   time_step: 9
// YAML:                   invalid_iterations: 2
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 49
// YAML:                   time_step: 6
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "EAST"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "GRANT_PREDICATE"
// YAML:                   id: 55
// YAML:                   time_step: 7
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 42
// YAML:                   time_step: 7
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:     - column: 2
// YAML:       row: 2
// YAML:       core_id: "10"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 1
// YAML:               operations:
// YAML:                 - opcode: "DATA_MOV"
// YAML:                   id: 490001
// YAML:                   time_step: 5
// YAML:                   invalid_iterations: 1
// YAML:                   src_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "WEST"
// YAML:                       color: "RED"
// YAML:     - column: 0
// YAML:       row: 3
// YAML:       core_id: "12"
// YAML:       entries:
// YAML:         - entry_id: "entry0"
// YAML:           instructions:
// YAML:             - index_per_ii: 2
// YAML:               operations:
// YAML:                 - opcode: "CONSTANT"
// YAML:                   id: 4
// YAML:                   time_step: 2
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "#0.000000"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:             - index_per_ii: 3
// YAML:               operations:
// YAML:                 - opcode: "GRANT_ONCE"
// YAML:                   id: 19
// YAML:                   time_step: 3
// YAML:                   invalid_iterations: 0
// YAML:                   src_operands:
// YAML:                     - operand: "$0"
// YAML:                       color: "RED"
// YAML:                   dst_operands:
// YAML:                     - operand: "SOUTH"
// YAML:                       color: "RED"
// ASM: # Compiled II: 4
// ASM:      PE(0,0):
// ASM-NEXT: {
// ASM-NEXT: CONSTANT, [#0] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: GRANT_ONCE, [$0] -> [EAST, RED] (t=1, inv_iters=0)
// ASM-NEXT: DATA_MOV, [NORTH, RED] -> [$0] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: GRANT_PREDICATE, [$0], [NORTH, RED] -> [NORTH, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(1,0):
// ASM-NEXT: {
// ASM-NEXT: DATA_MOV, [NORTH, RED] -> [EAST, RED] (t=4, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: PHI_START, [WEST, RED], [EAST, RED] -> [NORTH, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM:      PE(2,0):
// ASM-NEXT: {
// ASM-NEXT: GRANT_PREDICATE, [WEST, RED], [NORTH, RED] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: PHI_START, [EAST, RED], [NORTH, RED] -> [NORTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(3,0):
// ASM-NEXT: {
// ASM-NEXT: CONSTANT, [#10] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: GRANT_ONCE, [$0] -> [WEST, RED] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM:      PE(0,1):
// ASM-NEXT: {
// ASM-NEXT: PHI_START, [$0], [SOUTH, RED] -> [$0], [SOUTH, RED] (t=4, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: FADD, [NORTH, RED], [$0] -> [NORTH, RED], [EAST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: CONSTANT, [#3.000000] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: DATA_MOV, [EAST, RED] -> [SOUTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT: GRANT_ONCE, [$0] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(1,1):
// ASM-NEXT: {
// ASM-NEXT: CONSTANT, [#1] -> [$0] (t=0, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: GRANT_ONCE, [$0] -> [$0] (t=1, inv_iters=0)
// ASM-NEXT: DATA_MOV, [EAST, RED] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: PHI_START, [$0], [EAST, RED] -> [$0], [EAST, RED] (t=2, inv_iters=0)
// ASM-NEXT: DATA_MOV, [WEST, RED] -> [NORTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT: ADD, [SOUTH, RED], [$0] -> [EAST, RED], [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: DATA_MOV, [EAST, RED] -> [NORTH, RED] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(2,1):
// ASM-NEXT: {
// ASM-NEXT: ICMP_SLT, [WEST, RED], [SOUTH, RED] -> [SOUTH, RED], [NORTH, RED], [WEST, RED], [$2], [$1], [EAST, RED] (t=4, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: GRANT_PREDICATE, [$0], [$2] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: GRANT_PREDICATE, [$0], [$1] -> [SOUTH, RED] (t=6, inv_iters=1)
// ASM-NEXT: DATA_MOV, [EAST, RED] -> [WEST, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT: DATA_MOV, [WEST, RED] -> [$0] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(3,1):
// ASM-NEXT: {
// ASM-NEXT: NOT, [WEST, RED] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM:      PE(0,2):
// ASM-NEXT: {
// ASM-NEXT: PHI_START, [NORTH, RED], [EAST, RED] -> [SOUTH, RED] (t=4, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: DATA_MOV, [SOUTH, RED] -> [EAST, RED] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM:      PE(1,2):
// ASM-NEXT: {
// ASM-NEXT: GRANT_PREDICATE, [$0], [SOUTH, RED] -> [$0] (t=8, inv_iters=2)
// ASM-NEXT: } (idx_per_ii=0)
// ASM-NEXT: {
// ASM-NEXT: RETURN, [$0] (t=9, inv_iters=2)
// ASM-NEXT: } (idx_per_ii=1)
// ASM-NEXT: {
// ASM-NEXT: DATA_MOV, [EAST, RED] -> [$0] (t=6, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT: GRANT_PREDICATE, [WEST, RED], [$0] -> [WEST, RED] (t=7, inv_iters=1)
// ASM-NEXT: DATA_MOV, [SOUTH, RED] -> [$0] (t=7, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=3)
// ASM:      PE(2,2):
// ASM-NEXT: {
// ASM-NEXT: DATA_MOV, [SOUTH, RED] -> [WEST, RED] (t=5, inv_iters=1)
// ASM-NEXT: } (idx_per_ii=1)
// ASM:      PE(0,3):
// ASM-NEXT: {
// ASM-NEXT: CONSTANT, [#0.000000] -> [$0] (t=2, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=2)
// ASM-NEXT: {
// ASM-NEXT: GRANT_ONCE, [$0] -> [SOUTH, RED] (t=3, inv_iters=0)
// ASM-NEXT: } (idx_per_ii=3)
