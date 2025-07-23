// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --generate-code
// RUN: FileCheck %s --input-file=test/codegenerate/generated-instructions.json --check-prefix=INST
// RUN: FileCheck %s --input-file=test/codegenerate/generated-instructions.asm --check-prefix=ASM

// INST: "functions": [
// INST: "CompiledII": 1
// INST: "ResMII": 1
// INST: "name": "test"
// INST: "tile_instructions": {
// INST: "Tile(0)": {
// INST: "id": 0
// INST: "instructions": [
// INST: "constant_value": "1.000000"
// INST: "dst_direction": "South"
// INST: "dst_tile": "(1,2)"
// INST: "opcode": "constant"
// INST: "operands": []
// INST: "src_direction": "Local"
// INST: "src_tile": "(1,1)"
// INST: "time_step": 0
// INST: "x": 1
// INST: "y": 1
// INST: "Tile(1)": {
// INST: "id": 1
// INST: "constant_value": "2.000000"
// INST: "dst_direction": "East"
// INST: "dst_tile": "(1,2)"
// INST: "opcode": "constant"
// INST: "src_tile": "(2,2)"
// INST: "x": 2
// INST: "y": 2
// INST: "Tile(2)": {
// INST: "id": 2
// INST: "dst_direction": "Local"
// INST: "dst_tile": "(1,2)"
// INST: "opcode": "data_mov"
// INST: "operands": [
// INST: "arith.constant"
// INST: "src_direction": "South"
// INST: "src_tile": "(1,1)"
// INST: "src_direction": "East"
// INST: "src_tile": "(2,2)"
// INST: "dst_direction": "East"
// INST: "dst_tile": "(0,2)"
// INST: "opcode": "fadd"
// INST: "operands": [
// INST: "neura.data_mov"
// INST: "neura.data_mov"
// INST: "src_direction": "Local"
// INST: "src_tile": "(1,2)"
// INST: "time_step": 1
// INST: "x": 1
// INST: "y": 2
// INST: "Tile(3)": {
// INST: "id": 3
// INST: "dst_direction": "Local"
// INST: "dst_tile": "(0,2)"
// INST: "opcode": "data_mov"
// INST: "operands": [
// INST: "neura.fadd"
// INST: "src_direction": "East"
// INST: "src_tile": "(1,2)"
// INST: "opcode": "return"
// INST: "operands": [
// INST: "neura.data_mov"
// INST: "src_direction": "Local"
// INST: "src_tile": "(0,2)"
// INST: "time_step": 2
// INST: "x": 0
// INST: "y": 2

// ASM: PE(0,2):
// ASM: {
// ASM:     Entry [East, R] => Once {
// ASM:         {
// ASM:             RETURN, [Local, R]
// ASM:             NOP
// ASM:         }
// ASM:     }
// ASM: }
// ASM: PE(1,1):
// ASM: {
// ASM:     Entry [] => Once {
// ASM:         {
// ASM:             CONSTANT, IMM[1.000000e+00] -> [South, R]
// ASM:         }
// ASM:     }
// ASM: }
// ASM: PE(1,2):
// ASM: {
// ASM:     Entry [East, R], [South, R] => Once {
// ASM:         {
// ASM:             FADD, [South, R], [East, R] -> [East, R]
// ASM:         }
// ASM:     }
// ASM: }
// ASM: PE(2,2):
// ASM: {
// ASM:     Entry [] => Once {
// ASM:         {
// ASM:             CONSTANT, IMM[2.000000e+00] -> [East, R]
// ASM:         }
// ASM:     }
// ASM: }

func.func @test() -> f32 attributes {CompiledII = 1 : i32, ResMII = 1 : i32, accelerator = "neura"} {
  %a = arith.constant {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} 1.000000e+00 : f32
  %b = arith.constant {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} 2.000000e+00 : f32
  %a_mov = "neura.data_mov"(%a) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 0 : i32}]} : (f32) -> f32
  %b_mov = "neura.data_mov"(%b) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 0 : i32}]} : (f32) -> f32
  %res = "neura.fadd"(%a_mov, %b_mov) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (f32, f32) -> !neura.data<f32, i1>
  %res_mov = "neura.data_mov"(%res) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  "neura.return"(%res_mov) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> ()
}
