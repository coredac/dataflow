// RUN: mlir-neura-opt %s -assign-accelerator -lower-llvm-to-neura -leverage-predicated-value -transform-ctrl-to-data-flow -insert-data-mov -map-to-accelerator -generate-code
// RUN: FileCheck %s --input-file=test/Generated_Code/generated-instructions.json -check-prefix=JSON
// RUN: FileCheck %s --input-file=test/Generated_Code/generated-instructions.asm -check-prefix=ASM

// Test function for code generation verification.
func.func @test() -> f32 {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %0 = arith.addf %cst, %cst_0 : f32
  return %0 : f32
}

// JSON-CHECK: {
// JSON-CHECK:   "functions": [
// JSON-CHECK:     {
// JSON-CHECK:       "CompiledII": 1,
// JSON-CHECK:       "ResMII": 1,
// JSON-CHECK:       "name": "test",
// JSON-CHECK:       "tile_instructions": {
// JSON-CHECK:         "Tile(0)": {
// JSON-CHECK:           "id": 0,
// JSON-CHECK:           "instructions": [
// JSON-CHECK:             {
// JSON-CHECK:               "constant_value": "1.000000",
// JSON-CHECK:               "dst_direction": "South",
// JSON-CHECK:               "dst_tile": "(1,2)",
// JSON-CHECK:               "opcode": "constant",
// JSON-CHECK:               "operands": [],
// JSON-CHECK:               "src_direction": "Local",
// JSON-CHECK:               "src_tile": "(1,1)",
// JSON-CHECK:               "time_step": 0
// JSON-CHECK:             }
// JSON-CHECK:           ],
// JSON-CHECK:           "x": 1,
// JSON-CHECK:           "y": 1
// JSON-CHECK:         },
// JSON-CHECK:         "Tile(1)": {
// JSON-CHECK:           "id": 1,
// JSON-CHECK:           "instructions": [
// JSON-CHECK:             {
// JSON-CHECK:               "constant_value": "2.000000",
// JSON-CHECK:               "dst_direction": "East",
// JSON-CHECK:               "dst_tile": "(1,2)",
// JSON-CHECK:               "opcode": "constant",
// JSON-CHECK:               "operands": [],
// JSON-CHECK:               "src_direction": "Local",
// JSON-CHECK:               "src_tile": "(2,2)",
// JSON-CHECK:               "time_step": 0
// JSON-CHECK:             }
// JSON-CHECK:           ],
// JSON-CHECK:           "x": 2,
// JSON-CHECK:           "y": 2
// JSON-CHECK:         },
// JSON-CHECK:         "Tile(2)": {
// JSON-CHECK:           "id": 2,
// JSON-CHECK:           "instructions": [
// JSON-CHECK:             {
// JSON-CHECK:               "dst_direction": "Local",
// JSON-CHECK:               "dst_tile": "(1,2)",
// JSON-CHECK:               "opcode": "data_mov",
// JSON-CHECK:               "operands": [
// JSON-CHECK:                 "arith.constant"
// JSON-CHECK:               ],
// JSON-CHECK:               "src_direction": "South",
// JSON-CHECK:               "src_tile": "(1,1)",
// JSON-CHECK:               "time_step": 0
// JSON-CHECK:             },
// JSON-CHECK:             {
// JSON-CHECK:               "dst_direction": "Local",
// JSON-CHECK:               "dst_tile": "(1,2)",
// JSON-CHECK:               "opcode": "data_mov",
// JSON-CHECK:               "operands": [
// JSON-CHECK:                 "arith.constant"
// JSON-CHECK:               ],
// JSON-CHECK:               "src_direction": "East",
// JSON-CHECK:               "src_tile": "(2,2)",
// JSON-CHECK:               "time_step": 0
// JSON-CHECK:             },
// JSON-CHECK:             {
// JSON-CHECK:               "dst_direction": "East",
// JSON-CHECK:               "dst_tile": "(0,2)",
// JSON-CHECK:               "opcode": "fadd",
// JSON-CHECK:               "operands": [
// JSON-CHECK:                 "neura.data_mov",
// JSON-CHECK:                 "neura.data_mov"
// JSON-CHECK:               ],
// JSON-CHECK:               "src_direction": "Local",
// JSON-CHECK:               "src_tile": "(1,2)",
// JSON-CHECK:               "time_step": 1
// JSON-CHECK:             }
// JSON-CHECK:           ],
// JSON-CHECK:           "x": 1,
// JSON-CHECK:           "y": 2
// JSON-CHECK:         },
// JSON-CHECK:         "Tile(3)": {
// JSON-CHECK:           "id": 3,
// JSON-CHECK:           "instructions": [
// JSON-CHECK:             {
// JSON-CHECK:               "dst_direction": "Local",
// JSON-CHECK:               "dst_tile": "(0,2)",
// JSON-CHECK:               "opcode": "data_mov",
// JSON-CHECK:               "operands": [
// JSON-CHECK:                 "neura.fadd"
// JSON-CHECK:               ],
// JSON-CHECK:               "src_direction": "East",
// JSON-CHECK:               "src_tile": "(1,2)",
// JSON-CHECK:               "time_step": 1
// JSON-CHECK:             },
// JSON-CHECK:             {
// JSON-CHECK:               "opcode": "return",
// JSON-CHECK:               "operands": [
// JSON-CHECK:                 "neura.data_mov"
// JSON-CHECK:               ],
// JSON-CHECK:               "src_direction": "Local",
// JSON-CHECK:               "src_tile": "(0,2)",
// JSON-CHECK:               "time_step": 2
// JSON-CHECK:             }
// JSON-CHECK:           ],
// JSON-CHECK:           "x": 0,
// JSON-CHECK:           "y": 2
// JSON-CHECK:         }
// JSON-CHECK:       }
// JSON-CHECK:     }
// JSON-CHECK:   ]
// JSON-CHECK: }

// ASM-CHECK: PE(0,2):
// ASM-CHECK: {
// ASM-CHECK:     Entry [East, R] => Once {
// ASM-CHECK:         {
// ASM-CHECK:             RETURN, [Local, R]
// ASM-CHECK:             NOP
// ASM-CHECK:         }
// ASM-CHECK:     }
// ASM-CHECK: }
// ASM-CHECK: 
// ASM-CHECK: PE(1,1):
// ASM-CHECK: {
// ASM-CHECK:     Entry [] => Once {
// ASM-CHECK:         {
// ASM-CHECK:             CONSTANT, IMM[1.000000e+00] -> [South, R]
// ASM-CHECK:         }
// ASM-CHECK:     }
// ASM-CHECK: }
// ASM-CHECK: 
// ASM-CHECK: PE(1,2):
// ASM-CHECK: {
// ASM-CHECK:     Entry [East, R], [South, R] => Once {
// ASM-CHECK:         {
// ASM-CHECK:             FADD, [South, R], [East, R] -> [East, R]
// ASM-CHECK:         }
// ASM-CHECK:     }
// ASM-CHECK: }
// ASM-CHECK: 
// ASM-CHECK: PE(2,2):
// ASM-CHECK: {
// ASM-CHECK:     Entry [] => Once {
// ASM-CHECK:         {
// ASM-CHECK:             CONSTANT, IMM[2.000000e+00] -> [East, R]
// ASM-CHECK:         }
// ASM-CHECK:     }
// ASM-CHECK: } 