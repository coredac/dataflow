// RUN: mlir-neura-opt %s \
// RUN:   --assign-accelerator \
// RUN:   --lower-llvm-to-neura \
// RUN:   --leverage-predicated-value \
// RUN:   --transform-ctrl-to-data-flow \
// RUN:   --insert-data-mov \
// RUN:   --map-to-accelerator="mapping-strategy=heuristic" \
// RUN:   --generate-code
// RU: FileCheck %s --input-file=generated-instructions.json --check-prefix=CHECK
// RU: FileCheck %s --input-file=generated-instructions.asm --check-prefix=ASM

// CHECK-DAG: "opcode": "constant"
// CHECK-DAG: "opcode": "data_mov"
// CHECK-DAG: "opcode": "fadd"
// CHECK-DAG: "opcode": "return"
// CHECK-DAG: "CompiledII": 1
// CHECK-DAG: "ResMII": 1
// CHECK-DAG: "dst_direction": "East"
// CHECK-DAG: "src_direction": "Local"

// ASM: PE(0,2):
// ASM: RETURN, [Local, R]
// ASM: PE(1,2):
// ASM: FADD, [South, R], [East, R] -> [East, R]
// ASM: PE(2,2):
// ASM: CONSTANT, IMM[2.000000e+00] -> [East, R]

func.func @test() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fadd" (%a, %b) : (f32, f32) -> f32
  return %res : f32
}