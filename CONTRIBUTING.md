# Contributing to Neura

Thanks for your interest in contributing to Neura project!

## Guide for Contributing

Please follow the steps below:
  1. Raise an issue on the problems/freatures you try to solve/add.
  2. Finish coding and add corresponding tests.
  3. Submit a pull request and require review from other contributors.
  4. Merge into the `main` branch.

## Code Format Style

We follow LLVM style. The format configuration is in the `.clang-format` file
at the repo root — all tools below read it automatically.

You can choose one of the two options listed below for code formatting.

### Option 1: VS Code

1. Install the following extensions:
   - [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
     — C/C++ language server and formatter.
2. Add the following to your workspace or user `settings.json`
   (`Ctrl+Shift+P` → `Preferences: Open Workspace Settings (JSON)`):

```json
{
  "[c]": {
    "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd",
    "editor.formatOnSave": true
  },
  "[cpp]": {
    "editor.defaultFormatter": "llvm-vs-code-extensions.vscode-clangd",
    "editor.formatOnSave": true
  }
}
```

clangd automatically picks up `.clang-format` from the repo root, so no
additional style configuration is needed.

> If you use the `ms-vscode.cpptools` extension instead of clangd, set
> `"C_Cpp.clang_format_style": "file"` so it reads `.clang-format` rather
> than using a hard-coded style.

### Option 2: Command line (`git clang-format`)

Before submitting a patch, format your changed lines against `main`:

```sh
# Format only the lines changed relative to main (recommended before commit).
$ git clang-format main

# Preview the diff without touching files.
$ git clang-format --diff main

# Restrict to specific directories.
$ git clang-format main -- lib/ tools/
```

### One-time full-repo reformat

If you need to reformat the entire codebase (e.g., after changing `.clang-format`):

```sh
git ls-files | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx|inc|td)$' | xargs -r clang-format -i
```