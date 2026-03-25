# Contributing to Neura

Thanks for your interest in contributing to Neura project!

## Guide for Contributing

Please follow the steps below:
  1. Raise an issue on the problems/freatures you try to solve/add.
  2. Finish coding and add corresponding tests.
  3. Submit a pull request and require review from other contributors.
  4. Merge into the `main` branch.

## Code Format Style
We follow LLVM style and use `git clang-format` for formatting.

Please keep `.clang-format` simple and close to upstream LLVM style.

```sh
$ git clang-format main
```

Common usage:

```sh
# Format changed lines against main in working tree.
$ git clang-format main

# Preview formatting diff only.
$ git clang-format --diff main

# Restrict to selected files.
$ git clang-format main -- lib/ tools/
```