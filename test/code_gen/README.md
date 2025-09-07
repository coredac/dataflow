# Code Generation Test File Documentation

## Overview

This test file `test_code_generate.mlir` tests the final code generation process of a simple loop program. The program implements an accumulation loop from 0 to 9, adding 3.0 in each iteration.

## Program Semantics

Implements:
1. Initialize `i = 0`, `acc = 0.0`
2. While `i < 10`
3. `acc += 3.0`, `i += 1`
4. Return `acc`

## YAML Format Description

The generated YAML file describes the configuration and instruction scheduling of a 4x4 CGRA array:

### Basic Structure
- `array_config`: Array configuration information
  - `columns: 4`, `rows: 4`: 4x4 processor array
  - `cores`: Detailed information for each processing core

### Core Configuration
Each core contains:
- `column`, `row`: Position coordinates in the array
- `core_id`: Unique identifier for the core
- `entries`: List of execution contexts for this core

### Instruction Format
Each `entry` contains:
- `entry_id` — the (single-instruction) context ID  
- `instructions` — a list with one item:
  - `opcode` — e.g. `CONSTANT`, `DATA_MOV`, `PHI`, `ICMP`, `FADD`, …
  - `timestep` — the per-tile cycle at which the instruction executes
  - `src_operands` — inputs
  - `dst_operands` — outputs

Operands encode:
- `#N` — immediate constant (e.g., `#10`)
- `$N` — local register (e.g., `$22`)
- `EAST | WEST | NORTH | SOUTH` — directional ports for inter-tile communication

Each operand also has a `color` field. Colors are primarily meaningful for **directional** operands (to visualize/track routing); for registers and immediates, the color can be ignored by consumers.

### YAML Example

The following example shows the YAML structure for a few tiles with their instructions:

```yaml
array_config:
  columns: 4
  rows: 4
  cores:
    - column: 0
      row: 0
      core_id: "0"
      entries:
        - entry_id: "entry0"
          instructions:
            - opcode: "CONSTANT"
              timestep: 0
              src_operands:
                - operand: "#0"
                  color: "RED"
              dst_operands:
                - operand: "EAST"
                  color: "RED"
        - entry_id: "entry1"
          instructions:
            - opcode: "CONSTANT"
              timestep: 1
              src_operands:
                - operand: "#10"
                  color: "RED"
              dst_operands:
                - operand: "EAST"
                  color: "RED"
    - column: 1
      row: 1
      core_id: "5"
      entries:
        - entry_id: "entry0"
          instructions:
            - opcode: "PHI"
              timestep: 2
              src_operands:
                - operand: "EAST"
                  color: "RED"
                - operand: "SOUTH"
                  color: "RED"
              dst_operands:
                - operand: "EAST"
                  color: "RED"
        - entry_id: "entry1"
          instructions:
            - opcode: "ICMP"
              timestep: 4
              src_operands:
                - operand: "EAST"
                  color: "RED"
                - operand: "SOUTH"
                  color: "RED"
              dst_operands:
                - operand: "EAST"
                  color: "RED"
                - operand: "NORTH"
                  color: "RED"
                - operand: "$22"
                  color: "RED"
                - operand: "$21"
                  color: "RED"
                - operand: "SOUTH"
                  color: "RED"
                - operand: "$20"
                  color: "RED"
    - column: 2
      row: 1
      core_id: "6"
      entries:
        - entry_id: "entry0"
          instructions:
            - opcode: "ADD"
              timestep: 3
              src_operands:
                - operand: "WEST"
                  color: "RED"
                - operand: "SOUTH"
                  color: "RED"
              dst_operands:
                - operand: "WEST"
                  color: "RED"
                - operand: "$25"
                  color: "RED"
        - entry_id: "entry1"
          instructions:
            - opcode: "FADD"
              timestep: 5
              src_operands:
                - operand: "$24"
                  color: "RED"
                - operand: "NORTH"
                  color: "RED"
              dst_operands:
                - operand: "$26"
                  color: "RED"
                - operand: "EAST"
                  color: "RED"
```

#### What Each Tile Does:

**PE(0,0) - Constant Generation Tile:**
- `entry0` @t=0: Generates constant value 0 and sends it eastward
- `entry1` @t=1: Generates constant value 10 (loop upper bound) and sends it eastward

**PE(1,1) - Control Flow Tile:**
- `entry0` @t=2: Merges data flows from east and south using PHI operation
- `entry1` @t=4: Performs integer comparison (i < 10), broadcasts result to multiple destinations including registers $22, $21, $20

**PE(2,1) - Arithmetic Tile:**
- `entry0` @t=3: Performs integer addition (i + 1) and stores result in register $25
- `entry1` @t=5: Performs floating-point addition (accumulator + 3.0) and stores result in register $26


## ASM Format Description
The ASM format is a human-readable assembly-style view per tile.

### Basics
- **PE(x,y)** — the tile coordinates
- **Format** — `OPCODE, [src …] -> [dst …] (t=TIMESTEP)`
- **Operand tokens**
  - `#N` — immediate (e.g., `#0`, `#10`, `#3.000000`)
  - `$N` — local register (e.g., `$20`, `$22`, `$25`)
  - `[operand]` — non-directional (reg/imm)
  - `[DIRECTION, COLOR]` — directional with routing color
    - e.g., `[EAST, RED]`, `[WEST, RED]`, `[NORTH, RED]`, `[SOUTH, RED]`

## Timing Execution Examples 
### PE(0,0)
- @t=0: `CONSTANT [#0] -> [EAST]` - Generate constant 0 and send to east
- @t=1: `CONSTANT [#10] -> [EAST]` - Generate constant 10 (loop upper bound) and send to east
- @t=4: `DATA_MOV [EAST] -> [NORTH]` - Receive data from east and forward to north

### PE(1,1)
- @t=2: `PHI [EAST], [SOUTH] -> [EAST]` - Merge data flows from east and south
- @t=4: `ICMP [EAST], [SOUTH] -> [EAST], [NORTH], [$22], [$21], [SOUTH], [$20]` - Integer Compare operation, broadcasting results to multiple targets
- @t=5: `NOT [$20] -> [EAST]` - Negate the value in register $20
- @t=6: `GRANT_PREDICATE [WEST], [$21] -> [EAST]` - Data authorization based on predicate conditions
- @t=7: `CTRL_MOV [WEST] -> [$20]` - Control flow movement, updating register $20

### PE(2,1)
- @t=3: `ADD [WEST], [SOUTH] -> [WEST], [$25]` - Execute addition operation
- @t=4: `PHI [$24], [EAST] -> [$24]` - Data flow merging
- @t=5: `FADD [$24], [NORTH] -> [$26], [EAST]` - Execute floating-point addition
- @t=6: `GRANT_PREDICATE [$25], [$24] -> [WEST]` - Data authorization based on conditions

## Notes / Known Limitations

### Current Implementation Constraints
- **One instruction per entry**: Multiple operations within a tile are currently emitted as separate entries to satisfy the simulator requirements
- **Default color scheme**: You'll typically see "RED" as the default virtual channel color in the YAML output
- **Entry-based scheduling**: Each execution context (entry) can only contain one instruction at a time
