# Project Context: ResourceAwareTaskOptimization & CGRA Mapping Debugging

## Background
We are developing the `ResourceAwareTaskOptimizationPass` for the Taskflow-to-CGRA MLIR pipeline. The goal of this pass is to allocate CGRA tile counts to different taskflow tasks, profile their execution metrics (II, latency, res_mii, rec_mii), and fuse/balance them to optimize total pipeline latency over a limited CGRA resource budget (e.g., a 4x4 grid of CGRAs).

## Current Progress
- We fixed a crash that occurred related to block argument mismatch during fusion (`is_util_dummy` loop skipping in split-profiling).
- The pipeline now runs without crashing and correctly balances tasks up to the given CGRA limits.
- We modified `ii` attribute naming to `compiled_ii` per user's request.

## The Bug / The Next Step
The user reported the following critical issue:
> "还是不对，compiled ii只要等于1就肯定不对，更不可能小于res rec"
> (Translation: "Still wrong, it's definitely incorrect if compiled_ii equals 1, and it's even more impossible for it to be less than res_mii and rec_mii").

### Observation
In the emitted IR, we see task attributes like:
`{cgra_count = 2 : i32, compiled_ii = 1 : i64, rec_mii = 4 : i64, res_mii = 3 : i64, steps = 15 : i64, trip_count = 192 : i64}`

Here `compiled_ii` is `1`, which is logically impossible because it's smaller than the structural constraints `res_mii` (3) and `rec_mii` (4). `compiled_ii` should be at least `max(res_mii, rec_mii)`.

### Root Cause Suspects
This issue typically originates from one of these flow anomalies in `ResourceAwareTaskOptimizationPass.cpp`:

1.  **Fallback to default `1` inside `profileTask` or `reprofileForCgraCount`**: 
    If `Phase 1` lowering (`ConstructHyperblock`, `TaskflowToNeura`, etc.) fails or returns `kernels.empty()`, the function returns `node->ii` which sometimes falls back to a default `1` instead of dynamically taking `max(res_mii, rec_mii)`.
    
    *Look at `reprofileForCgraCount`*:
    ```cpp
    if (failed(pm1.run(phase1_module))) {
      return node->ii; // Often defaults to 1 for tmp_nodes!
    }
    // ...
    if (kernels.empty()) {
      return node->ii; // Defaults to 1! Should it analyze the loop at all?
    }
    ```

2.  **`compiled_ii` overrides during failure in `runNeuraPipelineOnKernel`**:
    If Mapper fails or completes, does it correctly populate `compiled_ii`? If Mapper doesn't explicitly rewrite the `mapping_info` attribute with `compiled_ii`, or if the retrieval fails to find `kCompiledII` (which you should verify is actually parsing correctly, see `MapToAcceleratorPass` emission), it might leave `compiled_ii` untouched, which could be 1 if it wasn't robustly bounded by `res_mii/rec_mii` internally.

3.  **Renaming `ii` to `compiled_ii` mapping issue**:
    I executed an automated `sed -i` to replace `"ii"` with `"compiled_ii"` in `ResourceAwareTaskOptimizationPass.cpp`. Make sure that any explicit initialization logic where `task->hasAttr("compiled_ii")` is handled correctly. Also, check where `node->ii` values are initially assigned. If a node is created and populated with `1`, it will persistently propagate `1` through fallback returns.

## Goal for Next Agent
1. Read `/tmp/agent_context.md` (this file).
2. Trace where `compiled_ii` = `1` is coming from in `ResourceAwareTaskOptimizationPass.cpp` and fix the fallback/initialization logic so that `compiled_ii` is strictly bounded by `max(res_mii, rec_mii)` even when inner kernels fail to map or lower.
3. Verify that the task metrics in `test/multi-cgra/taskflow/**/*.mlir` tests are logically sound (`compiled_ii >= res_mii && compiled_ii >= rec_mii`).
4. Re-run `ninja check-neura` to ensure tests compile and pass accurately with the correct `compiled_ii`.
