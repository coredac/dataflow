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

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically format code and check files before committing. This ensures consistent code style and catches common issues early.

### Setup

Install pre-commit and set up the git hooks:

```sh
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-commit
```

### What it does

The pre-commit hooks will automatically:
- Format C/C++ code using `clang-format` (with the project's `.clang-format` configuration)
- Remove trailing whitespace
- Ensure files end with a newline
- Validate YAML syntax
- Prevent committing large files

### Running manually

To run pre-commit on all files manually:

```sh
pre-commit run --all-files
```

Please ensure your code passes all pre-commit checks before submitting a pull request.