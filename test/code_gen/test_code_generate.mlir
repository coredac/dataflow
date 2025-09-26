// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --canonicalize-live-in \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --generate-code \
// RUN:   | FileCheck %s -check-prefix=MAPPING
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

// MAPPING:        func.func @loop_test() -> f32 attributes {accelerator = "neura", mapping_info = {compiled_ii = 6 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 4 : i32, res_mii = 1 : i32, x_tiles = 6 : i32, y_tiles = 6 : i32}} {
// MAPPING-NEXT:     %0 = "neura.constant"() <{predicate = true, value = 10 : i64}> {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %1 = "neura.data_mov"(%0) {mapping_locs = [{id = 56 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %2 = "neura.grant_once"(%1) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %3 = "neura.constant"() <{predicate = true, value = 0 : i64}> {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %4 = "neura.data_mov"(%3) {mapping_locs = [{id = 19 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %5 = "neura.grant_once"(%4) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %6 = "neura.constant"() <{predicate = true, value = 1 : i64}> {mapping_locs = [{id = 18 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<i64, i1>
// MAPPING-NEXT:     %7 = "neura.data_mov"(%6) {mapping_locs = [{id = 62 : i32, resource = "link", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %8 = "neura.grant_once"(%7) {mapping_locs = [{id = 24 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %9 = "neura.constant"() <{predicate = true, value = 3.000000e+00 : f32}> {mapping_locs = [{id = 19 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %10 = "neura.data_mov"(%9) {mapping_locs = [{id = 152 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %11 = "neura.grant_once"(%10) {mapping_locs = [{id = 19 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %12 = "neura.constant"() <{predicate = true, value = 0.000000e+00 : f32}> {mapping_locs = [{id = 18 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<f32, i1>
// MAPPING-NEXT:     %13 = "neura.data_mov"(%12) {mapping_locs = [{id = 144 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %14 = "neura.grant_once"(%13) {mapping_locs = [{id = 18 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %15 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %16 = "neura.data_mov"(%2) {mapping_locs = [{id = 56 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %17 = "neura.phi"(%15, %16) {mapping_locs = [{id = 7 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %18 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %19 = "neura.data_mov"(%8) {mapping_locs = [{id = 192 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %20 = "neura.phi"(%18, %19) {mapping_locs = [{id = 24 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 4 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %21 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %22 = "neura.data_mov"(%11) {mapping_locs = [{id = 152 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %23 = "neura.phi"(%21, %22) {mapping_locs = [{id = 19 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %24 = neura.reserve : !neura.data<f32, i1>
// MAPPING-NEXT:     %25 = "neura.data_mov"(%14) {mapping_locs = [{id = 62 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %26 = "neura.phi"(%24, %25) {mapping_locs = [{id = 24 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 4 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %27 = neura.reserve : !neura.data<i64, i1>
// MAPPING-NEXT:     %28 = "neura.data_mov"(%5) {mapping_locs = [{id = 48 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %29 = "neura.phi"(%27, %28) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %30 = "neura.data_mov"(%26) {mapping_locs = [{id = 82 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %31 = "neura.data_mov"(%23) {mapping_locs = [{id = 66 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %32 = "neura.fadd"(%30, %31) {mapping_locs = [{id = 25 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 4 : i32}]} : (!neura.data<f32, i1>, !neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %33 = "neura.data_mov"(%29) {mapping_locs = [{id = 18 : i32, resource = "link", time_step = 2 : i32}, {id = 40 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %34 = "neura.data_mov"(%20) {mapping_locs = [{id = 83 : i32, resource = "link", time_step = 2 : i32}, {id = 144 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %35 = "neura.add"(%33, %34) {mapping_locs = [{id = 18 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %36 = "neura.data_mov"(%35) {mapping_locs = [{id = 60 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %37 = "neura.data_mov"(%17) {mapping_locs = [{id = 22 : i32, resource = "link", time_step = 3 : i32}, {id = 44 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %38 = "neura.icmp"(%36, %37) <{cmpType = "slt"}> {mapping_locs = [{id = 19 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 3 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %39 = "neura.data_mov"(%35) {mapping_locs = [{id = 61 : i32, resource = "link", time_step = 4 : i32}, {id = 96 : i32, resource = "register", time_step = 5 : i32}, {id = 96 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %40 = "neura.data_mov"(%38) {mapping_locs = [{id = 63 : i32, resource = "link", time_step = 5 : i32}, {id = 61 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %41 = neura.grant_predicate %39, %40 {mapping_locs = [{id = 12 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %41 -> %27 {mapping_locs = [{id = 39 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %42 = "neura.data_mov"(%32) {mapping_locs = [{id = 200 : i32, resource = "register", time_step = 5 : i32}, {id = 200 : i32, resource = "register", time_step = 6 : i32}, {id = 200 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %43 = "neura.data_mov"(%38) {mapping_locs = [{id = 155 : i32, resource = "register", time_step = 5 : i32}, {id = 66 : i32, resource = "link", time_step = 6 : i32}, {id = 201 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %44 = neura.grant_predicate %42, %43 {mapping_locs = [{id = 25 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 4 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %44 -> %24 {mapping_locs = [{id = 85 : i32, resource = "link", time_step = 8 : i32}, {id = 192 : i32, resource = "register", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %45 = "neura.data_mov"(%23) {mapping_locs = [{id = 65 : i32, resource = "link", time_step = 4 : i32}, {id = 105 : i32, resource = "register", time_step = 5 : i32}, {id = 105 : i32, resource = "register", time_step = 6 : i32}, {id = 105 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %46 = "neura.data_mov"(%38) {mapping_locs = [{id = 154 : i32, resource = "register", time_step = 5 : i32}, {id = 65 : i32, resource = "link", time_step = 6 : i32}, {id = 104 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %47 = neura.grant_predicate %45, %46 {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     neura.ctrl_mov %47 -> %21 {mapping_locs = [{id = 44 : i32, resource = "link", time_step = 8 : i32}, {id = 153 : i32, resource = "register", time_step = 9 : i32}]} : !neura.data<f32, i1> !neura.data<f32, i1>
// MAPPING-NEXT:     %48 = "neura.data_mov"(%20) {mapping_locs = [{id = 82 : i32, resource = "link", time_step = 2 : i32}, {id = 87 : i32, resource = "link", time_step = 3 : i32}, {id = 152 : i32, resource = "register", time_step = 4 : i32}, {id = 152 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %49 = "neura.data_mov"(%38) {mapping_locs = [{id = 153 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %50 = neura.grant_predicate %48, %49 {mapping_locs = [{id = 19 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %50 -> %18 {mapping_locs = [{id = 63 : i32, resource = "link", time_step = 6 : i32}, {id = 62 : i32, resource = "link", time_step = 7 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %51 = "neura.data_mov"(%17) {mapping_locs = [{id = 56 : i32, resource = "register", time_step = 3 : i32}, {id = 22 : i32, resource = "link", time_step = 4 : i32}, {id = 104 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
// MAPPING-NEXT:     %52 = "neura.data_mov"(%38) {mapping_locs = [{id = 65 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %53 = neura.grant_predicate %51, %52 {mapping_locs = [{id = 13 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
// MAPPING-NEXT:     neura.ctrl_mov %53 -> %15 {mapping_locs = [{id = 43 : i32, resource = "link", time_step = 6 : i32}, {id = 57 : i32, resource = "register", time_step = 7 : i32}, {id = 57 : i32, resource = "register", time_step = 8 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
// MAPPING-NEXT:     %54 = "neura.data_mov"(%38) {mapping_locs = [{id = 66 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %55 = "neura.not"(%54) {mapping_locs = [{id = 25 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 4 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %56 = "neura.data_mov"(%32) {mapping_locs = [{id = 87 : i32, resource = "link", time_step = 5 : i32}, {id = 152 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     %57 = "neura.data_mov"(%55) {mapping_locs = [{id = 87 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
// MAPPING-NEXT:     %58 = neura.grant_predicate %56, %57 {mapping_locs = [{id = 19 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<f32, i1>, !neura.data<i1, i1> -> !neura.data<f32, i1>
// MAPPING-NEXT:     %59 = "neura.data_mov"(%58) {mapping_locs = [{id = 64 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
// MAPPING-NEXT:     "neura.return"(%59) {mapping_locs = [{id = 20 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<f32, i1>) -> ()
// MAPPING-NEXT:   }

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

// YAML:      array_config:
// YAML-NEXT:   columns: 6
// YAML-NEXT:   rows: 6
// YAML-NEXT:   cores:
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "6"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$48"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$48"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 1
// YAML-NEXT:       core_id: "7"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#10"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$56"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$56"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$56"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$56"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$57"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "12"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$96"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$96"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 2
// YAML-NEXT:       core_id: "13"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$104"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$105"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$104"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$104"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$105"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$104"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "18"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 0
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#1"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#0.000000"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$144"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$144"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$144"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ADD"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$144"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "19"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CONSTANT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "#3.000000"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "ICMP"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$153"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$153"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 7
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$152"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$153"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 2
// YAML-NEXT:       row: 3
// YAML-NEXT:       core_id: "20"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "RETURN"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 0
// YAML-NEXT:       row: 4
// YAML-NEXT:       core_id: "24"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 1
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_ONCE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$192"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 2
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$192"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 4
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "PHI"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "EAST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 9
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "CTRL_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$192"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:     - column: 1
// YAML-NEXT:       row: 4
// YAML-NEXT:       core_id: "25"
// YAML-NEXT:       entries:
// YAML-NEXT:         - entry_id: "entry0"
// YAML-NEXT:           instructions:
// YAML-NEXT:             - timestep: 3
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 5
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "FADD"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$200"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                 - opcode: "DATA_MOV"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "NORTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "$201"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 6
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "NOT"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "SOUTH"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:             - timestep: 8
// YAML-NEXT:               operations:
// YAML-NEXT:                 - opcode: "GRANT_PREDICATE"
// YAML-NEXT:                   src_operands:
// YAML-NEXT:                     - operand: "$200"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                     - operand: "$201"
// YAML-NEXT:                       color: "RED"
// YAML-NEXT:                   dst_operands:
// YAML-NEXT:                     - operand: "WEST"
// YAML-NEXT:                       color: "RED"


// ASM-LABEL: PE(0,1):
// ASM: {
// ASM:   GRANT_ONCE, [EAST, RED] -> [$48]
// ASM: } (t=1)
// ASM: {
// ASM:   PHI, [NORTH, RED], [$48] -> [NORTH, RED]
// ASM: } (t=2)

// ASM-LABEL: PE(1,1):
// ASM: {
// ASM:   CONSTANT, [#0] -> [WEST, RED]
// ASM: } (t=0)
// ASM: {
// ASM:   CONSTANT, [#10] -> [$56]
// ASM: } (t=1)
// ASM: {
// ASM:   GRANT_ONCE, [$56] -> [$56]
// ASM: } (t=2)
// ASM: {
// ASM:   PHI, [NORTH, RED], [$56] -> [NORTH, RED]
// ASM: } (t=3)
// ASM: {
// ASM:   CTRL_MOV, [SOUTH, RED] -> [$57]
// ASM: } (t=7)

// ASM-LABEL: PE(0,2):
// ASM: {
// ASM:   DATA_MOV, [SOUTH, RED] -> [NORTH, RED]
// ASM: } (t=3)
// ASM: {
// ASM:   DATA_MOV, [SOUTH, RED] -> [$96]
// ASM: } (t=5)
// ASM: {
// ASM:   GRANT_PREDICATE, [$96], [NORTH, RED] -> [SOUTH, RED]
// ASM: } (t=7)

// ASM-LABEL: PE(1,2):
// ASM: {
// ASM:   DATA_MOV, [NORTH, RED] -> [$104]
// ASM: } (t=3)
// ASM: {
// ASM:   DATA_MOV, [SOUTH, RED] -> [NORTH, RED]
// ASM: } (t=4)
// ASM: {
// ASM:   DATA_MOV, [SOUTH, RED] -> [$105]
// ASM:   DATA_MOV, [SOUTH, RED] -> [$104]
// ASM: } (t=5)
// ASM: {
// ASM:   GRANT_PREDICATE, [$104], [NORTH, RED] -> [SOUTH, RED]
// ASM: } (t=6)
// ASM: {
// ASM:   GRANT_PREDICATE, [$105], [$104] -> [NORTH, RED]
// ASM: } (t=8)

// ASM-LABEL: PE(0,3):
// ASM: {
// ASM:   CONSTANT, [#1] -> [NORTH, RED]
// ASM: } (t=0)
// ASM: {
// ASM:   CONSTANT, [#0.000000] -> [$144]
// ASM: } (t=2)
// ASM: {
// ASM:   GRANT_ONCE, [$144] -> [NORTH, RED]
// ASM:   DATA_MOV, [SOUTH, RED] -> [$144]
// ASM: } (t=3)
// ASM: {
// ASM:   ADD, [SOUTH, RED], [$144] -> [EAST, RED], [SOUTH, RED]
// ASM: } (t=4)
// ASM: {
// ASM:   DATA_MOV, [EAST, RED] -> [SOUTH, RED]
// ASM: } (t=6)
// ASM: {
// ASM:   CTRL_MOV, [EAST, RED] -> [NORTH, RED]
// ASM: } (t=7)

// ASM-LABEL: PE(1,3):
// ASM: {
// ASM:   CONSTANT, [#3.000000] -> [$152]
// ASM: } (t=2)
// ASM: {
// ASM:   GRANT_ONCE, [$152] -> [$152]
// ASM: } (t=3)
// ASM: {
// ASM:   PHI, [SOUTH, RED], [$152] -> [NORTH, RED], [SOUTH, RED]
// ASM:   DATA_MOV, [SOUTH, RED] -> [$152]
// ASM: } (t=4)
// ASM: {
// ASM:   ICMP, [WEST, RED], [SOUTH, RED] -> [WEST, RED], [NORTH, RED], [SOUTH, RED], [$153]
// ASM: } (t=5)
// ASM: {
// ASM:   GRANT_PREDICATE, [$152], [$153] -> [WEST, RED]
// ASM:   DATA_MOV, [SOUTH, RED] -> [$152]
// ASM: } (t=6)
// ASM: {
// ASM:   GRANT_PREDICATE, [$152], [NORTH, RED] -> [EAST, RED]
// ASM: } (t=7)
// ASM: {
// ASM:   CTRL_MOV, [NORTH, RED] -> [$153]
// ASM: } (t=9)

// ASM-LABEL: PE(2,3):
// ASM: {
// ASM:   RETURN, [WEST, RED]
// ASM: } (t=8)

// ASM-LABEL: PE(0,4):
// ASM: {
// ASM:   GRANT_ONCE, [SOUTH, RED] -> [$192]
// ASM: } (t=1)
// ASM: {
// ASM:   PHI, [SOUTH, RED], [$192] -> [SOUTH, RED], [EAST, RED]
// ASM: } (t=2)
// ASM: {
// ASM:   PHI, [EAST, RED], [SOUTH, RED] -> [EAST, RED]
// ASM: } (t=4)
// ASM: {
// ASM:   CTRL_MOV, [WEST, RED] -> [$192]
// ASM: } (t=9)

// ASM-LABEL: PE(1,4):
// ASM: {
// ASM:   DATA_MOV, [WEST, RED] -> [SOUTH, RED]
// ASM: } (t=3)
// ASM: {
// ASM:   FADD, [WEST, RED], [SOUTH, RED] -> [$200], [SOUTH, RED]
// ASM:   DATA_MOV, [NORTH, RED] -> [$201]
// ASM: } (t=5)
// ASM: {
// ASM:   NOT, [SOUTH, RED] -> [SOUTH, RED]
// ASM: } (t=6)
// ASM: {
// ASM:   GRANT_PREDICATE, [$200], [$201] -> [WEST, RED]
// ASM: } (t=8)
