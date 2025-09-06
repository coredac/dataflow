# Contributing to Neura

Thanks for your interest in contributing to Neura project!

## Guide for Contributing

Please follow the steps below:
  1. Raise an issue on the problems/freatures you try to solve/add.
  2. Finish coding and add corresponding tests.
  3. Submit a pull request and require review from other contributors.
  4. Merge into the `main` branch.

## Code Format Style
We take the LLVM style as the code format.

To follow this style, you can use `clang-format` to generate the format configuration file.

```sh
$ clang-format -style=LLVM -dump-config > .clang-format
```

Then use some format extentions in your editor to enable automatic code formatting (e.g., `clangd` in VS Code).