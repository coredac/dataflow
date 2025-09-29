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
- `entry_id` — the execution context ID (typically "entry0" and can be extended to multiple entries in the future )
- `instructions` — a list of instruction groups organized by timestep:
  - `timestep` — the per-tile cycle at which the instruction group executes
  - `operations` — a list of operations executed in this timestep
    - `opcode` — e.g. `CONSTANT`, `DATA_MOV`, `PHI`, `ICMP`, `FADD`, `GRANT_ONCE`, `GRANT_PREDICATE`, `CTRL_MOV`, `NOT`, `RETURN`
    - `src_operands` — inputs
    - `dst_operands` — output

Operands encode:
- `#N` — immediate constant (e.g., `#10`)
- `$N` — local register (e.g., `$22`)
- `EAST | WEST | NORTH | SOUTH` — directional ports for inter-tile communication

Each operand also has a `color` field. Colors are primarily meaningful for **directional** operands (to visualize/track routing), rather than register operands

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
            - timestep: 0
              operations:
                - opcode: "CONSTANT"
                  src_operands:
                    - operand: "#0"
                      color: "RED"
                  dst_operands:
                    - operand: "EAST"
                      color: "RED"
            - timestep: 1
              operations:
                - opcode: "CONSTANT"
                  src_operands:
                    - operand: "#10"
                      color: "RED"
                  dst_operands:
                    - operand: "EAST"
                      color: "RED"
            - timestep: 4
              operations:
                - opcode: "DATA_MOV"
                  src_operands:
                    - operand: "EAST"
                      color: "RED"
                  dst_operands:
                    - operand: "NORTH"
                      color: "RED"
    - column: 1
      row: 0
      core_id: "1"
      entries:
        - entry_id: "entry0"
          instructions:
            - timestep: 1
              operations:
                - opcode: "GRANT_ONCE"
                  src_operands:
                    - operand: "WEST"
                      color: "RED"
                  dst_operands:
                    - operand: "NORTH"
                      color: "RED"
            - timestep: 2
              operations:
                - opcode: "GRANT_ONCE"
                  src_operands:
                    - operand: "WEST"
                      color: "RED"
                  dst_operands:
                    - operand: "$4"
                      color: "RED"
            - timestep: 3
              operations:
                - opcode: "PHI"
                  src_operands:
                    - operand: "$5"
                      color: "RED"
                    - operand: "$4"
                      color: "RED"
                  dst_operands:
                    - operand: "NORTH"
                      color: "RED"
                    - operand: "$4"
                      color: "RED"
                - opcode: "DATA_MOV"
                  src_operands:
                    - operand: "EAST"
                      color: "RED"
                  dst_operands:
                    - operand: "WEST"
                      color: "RED"
```

#### What Each Tile Does:

**PE(0,0) - Constant Generation Tile:**
- @t=0: Generates constant value 0 and sends it eastward
- @t=1: Generates constant value 10 (loop upper bound) and sends it eastward
- @t=4: Forwards data from east to north

**PE(1,0) - Control Flow Tile:**
- @t=1: Grants data from west to north
- @t=2: Grants data from west to register $4
- @t=3: Merges data flows using PHI operation and forwards data from east to west
- @t=5: Performs conditional data authorization based on predicate conditions


## ASM Format Description
The ASM format is a human-readable assembly-style view per tile.

### Basics
- **PE(x,y)** — the tile coordinates
- **Format** — Operations are grouped by timestep in `{}` blocks with `(t=TIMESTEP)` suffix
- **Operation format** — `OPCODE, [src …] -> [dst …]` or `OPCODE -> [dst …]` (for operations without source operands)
- **Operand tokens**
  - `#N` — immediate (e.g., `#0`, `#10`, `#3.000000`)
  - `$N` — local register (e.g., `$4`, `$5`, `$8`)
  - `[operand]` — non-directional (reg/imm)
  - `[DIRECTION, COLOR]` — directional with routing color
    - e.g., `[EAST, RED]`, `[WEST, RED]`, `[NORTH, RED]`, `[SOUTH, RED]`

## Timing Execution Examples 
### PE(0,0)
```
{
  CONSTANT, [#0] -> [EAST, RED]
} (t=0)
{
  CONSTANT, [#10] -> [EAST, RED]
} (t=1)
{
  DATA_MOV, [EAST, RED] -> [NORTH, RED]
} (t=4)
```

### PE(1,0)
```
{
  GRANT_ONCE, [WEST, RED] -> [NORTH, RED]
} (t=1)
{
  GRANT_ONCE, [WEST, RED] -> [$4]
} (t=2)
{
  PHI, [$5], [$4] -> [NORTH, RED], [$4]
  DATA_MOV, [EAST, RED] -> [WEST, RED]
} (t=3)
{
  GRANT_PREDICATE, [$4], [NORTH, RED] -> [$5]
} (t=5)
```

### PE(2,0)
```
{
  CONSTANT, [#1] -> [$8]
} (t=0)
{
  ADD, [WEST, RED], [SOUTH, RED] -> [WEST, RED], [$25]
} (t=3)
{
  FADD, [$24], [NORTH, RED] -> [$26], [EAST, RED]
} (t=5)
```

## Notes / Known Limitations

### Current Implementation Constraints
- **Timestep-based grouping**: Operations are grouped by timestep within each entry, allowing multiple operations to execute in the same cycle
- **Default color scheme**: You'll typically see "RED" as the default virtual channel color in the YAML output
- **Entry-based scheduling**: Each execution context (entry) can contain multiple instructions organized by timestep
