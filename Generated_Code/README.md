# Generated Code Files

This folder contains the generated code files from the Neura dialect compilation pipeline.

## Files Description

### Input File
- **test.mlir**: Original MLIR test file containing a simple function with two constants and an addition operation.

### Generated Output Files
- **generated-instructions.json**: JSON format output containing detailed instruction information organized by Processing Elements (PEs), including:
  - PE coordinates and IDs
  - Instruction opcodes and operands
  - Source and destination directions for data movement
  - Time steps for each instruction
  - Constant values

- **generated-instructions.asm**: Assembly-like format output containing:
  - PE-specific code blocks with Entry conditions
  - Data flow directions in [direction, color] format
  - Operation sequences with input/output directions
  - Loop/Once block structure

## Test Case Details

The test case implements a simple function:
```mlir
func.func @test() -> f32 {
  %a = arith.constant 1.0 : f32
  %b = arith.constant 2.0 : f32
  %res = "neura.fadd" (%a, %b) : (f32, f32) -> f32
  return %res : f32
}
```

## Generated Architecture Mapping

The compilation pipeline maps the operations to a spatial architecture:

- **PE(1,1)**: Generates constant 1.0, sends to North
- **PE(2,2)**: Generates constant 2.0, sends to West  
- **PE(1,2)**: Receives data from North and West, performs FADD, sends result to West
- **PE(0,2)**: Receives data from West, performs RETURN

## Data Flow Directions

The generated code correctly calculates data movement directions:
- North: Data moving from lower Y to higher Y
- West: Data moving from higher X to lower X
- Local: Data staying within the same PE

## Usage

These files were generated using the command:
```bash
./build/tools/mlir-neura-opt/mlir-neura-opt test/test.mlir \
  -pass-pipeline="builtin.module(assign-accelerator,lower-llvm-to-neura,leverage-predicated-value,transform-ctrl-to-data-flow,insert-data-mov,map-to-accelerator,generate-code)" \
  -o /dev/null
```

The GenerateCodePass automatically creates both JSON and ASM output files in the current directory. 