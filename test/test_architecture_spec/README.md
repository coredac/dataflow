# Test Architecture Specification

This directory contains a specialized 4x4 CGRA (Coarse-Grained Reconfigurable Array) architecture specification designed specifically for testing purposes.

## Overview

The `architecture.yaml` file defines a simplified 4x4 NeuraCGRA architecture that provides a consistent testing environment for various test suites including:

- `code_gen` - Code generation tests
- `mapping_quality` - Mapping quality evaluation tests  
- `neura` - Neura dialect tests
- `controflow_fuse` - Control flow fusion tests

## Architecture Configuration

### Key Features

- **Array Size**: 4x4 CGRA (16 processing elements)
- **Registers**: 32 registers per tile (4 register files × 8 registers each)
- **Operations**: Comprehensive set of operations including arithmetic, logical, control flow, and data movement operations
- **Connectivity**: Standard mesh topology
- **No Overrides**: Clean configuration without link or tile overrides for consistent testing

### Register Allocation

Each tile has 32 registers allocated as follows:
- Tile 0: registers 0-31
- Tile 1: registers 32-63  
- Tile 2: registers 64-95
- And so on...

This allocation scheme ensures predictable register numbering in test outputs.

## Usage

To use this architecture specification in your tests, add the following option to your `mlir-neura-opt` command:

```bash
--architecture-spec=test_architecture_spec/architecture.yaml
```

### Example

```bash
mlir-neura-opt input.mlir \
  --assign-accelerator \
  --lower-llvm-to-neura \
  --map-to-accelerator="mapping-strategy=heuristic" \
  --architecture-spec=test_architecture_spec/architecture.yaml \
  --generate-code
```

## Path Considerations

When using this architecture specification from different test directories, use the appropriate relative path:

- From `test/`: `test_architecture_spec/architecture.yaml`
- From `test/code_gen/`: `../test_architecture_spec/architecture.yaml`  
- From `test/neura/ctrl/`: `../../test_architecture_spec/architecture.yaml`

## Benefits

1. **Consistency**: All tests use the same 4x4 architecture specification
2. **Predictability**: Fixed register allocation ensures stable test outputs
3. **Simplicity**: No complex overrides or custom configurations
4. **Portability**: Works across different test environments and CI systems
