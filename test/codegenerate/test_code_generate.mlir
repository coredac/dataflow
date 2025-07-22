// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --generate-code
// RUN: FileCheck %s --input-file=test/codegenerate/generated-instructions.json --check-prefix=CHECK-DAG
// RUN: FileCheck %s --input-file=test/codegenerate/generated-instructions.asm --check-prefix=ASM

// CHECK-DAG: "opcode": "constant"
// CHECK-DAG: "opcode": "data_mov"
// CHECK-DAG: "opcode": "fadd"
// CHECK-DAG: "src_direction": "Local"
// CHECK-DAG: "dst_direction": "East"
// CHECK-DAG: "dst_direction": "South"
// CHECK-DAG: "time_step": 0
// CHECK-DAG: "time_step": 1
// CHECK-DAG: "time_step": 2
// CHECK-DAG: "x": 1
// CHECK-DAG: "y": 2
// CHECK-DAG: "CompiledII": 1
// CHECK-DAG: "ResMII": 1

// ASM: PE(0,2):
// ASM: RETURN, [Local, R]
// ASM: NOP
// ASM: PE(1,1):
// ASM: CONSTANT, IMM[1.000000e+00] -> [South, R]
// ASM: PE(1,2):
// ASM: FADD, [South, R], [East, R] -> [East, R]
// ASM: PE(2,2):
// ASM: CONSTANT, IMM[2.000000e+00] -> [East, R]
// ASM-DAG: Entry [] => Once {

func.func @test() -> f32 attributes {CompiledII = 1 : i32, ResMII = 1 : i32, accelerator = "neura"} {
  %a = arith.constant {mapping_locs = [{id = 5 : i32, resource = "tile", time_step = 0 : i32, x = 1 : i32, y = 1 : i32}]} 1.000000e+00 : f32
  %b = arith.constant {mapping_locs = [{id = 10 : i32, resource = "tile", time_step = 0 : i32, x = 2 : i32, y = 2 : i32}]} 2.000000e+00 : f32
  %a_mov = "neura.data_mov"(%a) {mapping_locs = [{id = 16 : i32, resource = "link", time_step = 0 : i32}]} : (f32) -> f32
  %b_mov = "neura.data_mov"(%b) {mapping_locs = [{id = 31 : i32, resource = "link", time_step = 0 : i32}]} : (f32) -> f32
  %res = "neura.fadd"(%a_mov, %b_mov) {mapping_locs = [{id = 6 : i32, resource = "tile", time_step = 1 : i32, x = 1 : i32, y = 2 : i32}]} : (f32, f32) -> !neura.data<f32, i1>
  %res_mov = "neura.data_mov"(%res) {mapping_locs = [{id = 17 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<f32, i1>) -> !neura.data<f32, i1>
  "neura.return"(%res_mov) {mapping_locs = [{id = 2 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<f32, i1>) -> ()
}
