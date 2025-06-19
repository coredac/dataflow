# Tests for Neura

The structure of the files in this folder is as follows：
```
.
├── affine2neura
│   └── bert
├── arith2neura
│   ├── add.mlir
│   └── Output
├── c2llvm2mlir
│   ├── kernel.cpp
│   ├── Output
│   └── test.mlir
├── lit.cfg
├── lit.cfg.in
├── neura
│   ├── arith_add.mlir
│   ├── ctrl
│   ├── fadd_fadd.mlir
│   ├── for_loop
│   ├── interpreter
│   ├── llvm_add.mlir
│   ├── llvm_sub.mlir
│   └── Output
├── Output
│   └── test.mlir.script
├── README.md
├── samples
│   ├── bert
│   └── lenet
└── test.mlir
```

All of the above content can be divided into three categories

## 1 Conversion Test
We need to convert other dialects to our `neura` dialect for compilation optimization. In order to verify the correctness of conversions from other dialects to `nerua` dialect, we need to provide the appropriate test for a conversion pass from a dialect to `nerua` dialect.

For now, we have:
`affine2neura`: tests provided for `--lower-affine-to-neura` [To be provided]
`arith2neura`: tests provided for `--lower-arith-to-neura`
`c2llvm2mlir`: tests provided for `--lower-llvm-to-neura`

## 2 Neura Compiler Test
Tests for individual passes/pass pipelines at the `neura` dialect level.

## 3 Samples
A collection of real-world applications for generating unit small tests.

For now, [BERT](https://github.com/codertimo/BERT-pytorch) and [LENET](https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py) are included.

We generate the `linalg` dialect of these models via [Torch MLIR](https://github.com/llvm/torch-mlir). which is then lowered to `affine` dialect for further lowering.

Due to the data dependencies between loops in models, we are now unable to automatically extract each of these SINGLE loops from the model IR for individual tests.

But we can manually collect some small unit tests from these sample IRs. For example, you can write `c++` code of a loop from BERT by mimicing the its corresponding `affine.for` operations, then use [Polygeist](https://github.com/llvm/Polygeist) to convert these `c++` code into `affine` mlir for further lowering. And that's how we generated tests in `affine2neura/bert`.