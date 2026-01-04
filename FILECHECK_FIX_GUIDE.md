# Neura Dialect Test Fix Guide (FileCheck & Determinism)

This guide summarizes the troubleshooting steps and best practices developed during the systematic fixing of Dataflow mapping tests (`fir_kernel.mlir`, `branch_for.mlir`, etc.).

## 1. Handling Parallel Test Conflicts
**Problem:** In a parallel test environment (like `llvm-lit -j8`), tests that write to the same hardcoded temporary file name will crash each other.
**Evidence:** Random failures in `fir_kernel.mlir` and `fir_kernel_vec.mlir` when running the full suite, but passing when run individually.
**Cause:** `GenerateCodePass.cpp` hardcodes `tmp-generated-instructions.yaml` and `.asm`.
**Workaround:** Ensure tests are run in isolation during debugging or update the pass to accept a unique output prefix.

## 2. Debug Information Pollution
**Problem:** `[DEBUG]` logs or `Collecting recurrence cycles` messages appearing in standard output break `CHECK-NEXT` sequences.
**Solution:**
- Use `CHECK` instead of `CHECK-NEXT` at block boundaries (e.g., between instructions or different PEs).
- Explicitly pipe output through `grep -v "\[DEBUG\]"` in the `RUN` command if possible.
- Avoid matching unstable attributes like `res_mii` or `rec_mii` in `MAPPING` headers if they are prone to fluctuation. Match only the stable parts: `// MAPPING: func.func @name() ... attributes {accelerator = "neura"`.

## 3. Dealing with Non-Deterministic Off-by-One Mismatches
**Problem:** `FileCheck` reports a match on the "wrong line" even if the content looks identical.
**Reason:** Usually a hidden newline, an interleaved debug line, or an empty line that `CHECK-NEXT` cannot bridge.
**Best Practice:**
- For `ASM` blocks, use `// ASM:` (without `-NEXT`) for the first line of each PE block.
- Only use `// ASM-NEXT:` for lines *inside* a `{ ... }` block where no empty lines/debug info are expected.

## 4. Regeneration Heuristics
**Rule of Thumb:**
- Use `latest` IR only if it reflects the logic changes (e.g., II reduction).
- Truncate long `YAML` or `ASM` outputs to the first 25-50 lines to keep tests maintainable.
- Ensure the `RUN` line includes all necessary passes (`--assign-accelerator`, `--lower-llvm-to-neura`, etc.) in the exact order intended by the optimization pipeline.

## 5. YAML & ASM checks should not exceed 60 lines, but should preserve all pre-existing info like compiled_ii