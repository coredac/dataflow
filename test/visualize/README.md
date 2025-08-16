# MLIR Code Visualization Guide

This guide explains how to visualize MLIR (Multi-Level Intermediate Representation) code using various tools and techniques.

## Usage

First, run `apt install graphviz` to install Graphviz. For an MLIR file, use the option `--view-op-graph` to generate a Graphviz visualization of a function. Then, use the `dot` command to create an image file from the visualization.

Example:

```bash
mlir-neura-opt --assign-accelerator --lower-arith-to-neura --fuse-patterns test.mlir > test_fused.mlir
mlir-opt --allow-unregistered-dialect --view-op-graph test_fused.mlir 2> test.dot 
dot -Tpng test.dot -o test.png
```

It will generate a `test.png` file in the current directory, as shown below:

![test](test.png)

For a more complicated example, we can use `mlir-opt --allow-unregistered-dialect --view-op-graph test2.mlir` to visualize the code.

```bash
mlir-opt --lower-affine --convert-scf-to-cf --convert-cf-to-llvm test2.mlir -o test2_llvm.mlir
mlir-neura-opt test2_llvm.mlir --assign-accelerator --lower-arith-to-neura --lower-memref-to-neura --lower-builtin-to-neura --lower-llvm-to-neura --canonicalize-cast --canonicalize-live-in --leverage-predicated-value --transform-ctrl-to-data-flow > test2_neura.mlir 
mlir-opt --allow-unregistered-dialect --view-op-graph test2_neura.mlir 2> test2.dot 
dot -Tpng test2.dot -o test2.png
```

![test2](test2.png)