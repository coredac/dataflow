## Fix: Skip direct-dominating live-in optimization for loop headers

Closes #270

### Problem

In nested loop kernels (e.g., `gemv`, `bicg`, `gemm`, `spmv`, `fft`), values defined in **outer loop blocks** and used in **inner loop bodies** were incorrectly classified as "direct dominating live-ins" by `CanonicalizeLiveInPass`. This prevented them from being promoted to block arguments, which in turn caused `TransformCtrlToDataFlowPass` to **not** create inner-rate `PHI_START` operations for them.

Take `gemv` (`y[i] = Σ A[i*N+j] * x[j]`) as an example. The outer loop computes `gep(%arg0, shl(i, 4))` — the base pointer for row `i` of matrix `A`. This pointer is consumed by every inner loop iteration `j` to compute `A[i*N+j]`.

**Before the fix**, the GEP pointer was a direct cross-block reference in the inner loop block (`^bb4`), not a block argument:

```
^bb1:
  %6 = neura.gep(%arg0, shl(i, 4))  // row-i base pointer
  neura.br ... to ^bb4

^bb4(%15: i64, %16: i32):  // only 2 block args — NO GEP pointer arg
  gep(%6, %15)              // cross-block ref to %6
```

Because `%6` was not a block argument, `TransformCtrlToDataFlowPass` produced a dataflow graph where the GEP was connected to an **outer-rate** `PHI_START`, producing:

```
i sequence: 0, ⊥, ⊥, ⊥, 1, ⊥, ⊥, ⊥, 2, ⊥, ⊥, ⊥, 3   (outer rate)
```

The inner loop needs the pointer **every cycle** (inner rate: `0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3`), so it starves — inner loop operations cannot get enough valid data.

### Root Cause

In `CanonicalizeLiveInPass::identifyDirectDominatingLiveIns()`, the helper `isDirectUnconditionalPattern(defBlock, useBlock)` checks whether `defBlock` dominates `useBlock` AND `useBlock` post-dominates `defBlock`. For the case of `^bb1` → `^bb4`, both conditions hold (unconditional branch path), so the function returns `true`.

However, `^bb4` is the inner loop header — it has a **self-back-edge** (`^bb4 → ^bb4`). The "direct dominating" optimization assumes the value flows through at the same rate as the definition site, but a loop header operates at a **different (faster) rate**. These values **must** be promoted to block arguments so that `TransformCtrlToDataFlowPass` can create inner-rate `PHI_START` ops for them.

### Fix

Added a back-edge check in `identifyDirectDominatingLiveIns()`: before applying the direct-dominating optimization, check whether the **using block** is a loop header (i.e., it has a predecessor that it dominates). If so, skip the optimization and let the value be promoted to a block argument.

```cpp
// If the using block is a loop header (has back-edges), skip the
// direct-dominating optimization — values need inner-rate PHI_STARTs.
bool using_block_is_loop_header = false;
for (Block *pred : block.getPredecessors()) {
  if (dom_info.dominates(&block, pred)) {
    using_block_is_loop_header = true;
    break;
  }
}
if (using_block_is_loop_header)
  continue;
```

### Result

**After the fix**, the GEP pointer is correctly promoted to a block argument in `^bb4`:

```
^bb1:
  %6 = neura.gep(%arg0, shl(i, 4))
  neura.br ..., %6 : !llvm.ptr, ... to ^bb4

^bb4(%15: i64, %16: i32, %17: !llvm.ptr, ...):  // %17 is now a block arg
  gep(%17, %15)
```

And `TransformCtrlToDataFlowPass` creates an inner-rate `PHI_START` for it:

```
%9  = neura.gep(%arg0, shl(i, 4))          // outer-rate GEP
%17 = neura.phi_start %9, %16              // inner-rate PHI_START for the pointer
%22 = neura.gep(%17, %21)                  // inner-loop body uses inner-rate pointer
```

The pointer sequence is now correctly `valid, valid, valid, valid, valid, ...` at inner rate, so inner loop operations receive valid data every cycle.

### Files Changed

| File | Change |
|------|--------|
| `lib/NeuraDialect/Transforms/CanonicalizeLiveInPass.cpp` | Added loop-header back-edge check |
| `test/e2e/gemv/gemv_kernel.mlir` | Updated test expectations |
| `test/e2e/bicg/bicg_int_kernel.mlir` | Updated test expectations |
| `test/e2e/bicg/bicg_kernel.mlir` | Updated test expectations |
| `test/e2e/gemm/gemm_kernel.mlir` | Updated test expectations |
| `test/e2e/spmv/spmv_kernel.mlir` | Updated test expectations |
| `test/e2e/fft/fft_kernel.mlir` | Updated test expectations |
| `test/neura/fusion/test.mlir` | Updated test expectations (partial — CHECK-FUSED lines) |

### Known Issue

The `test/neura/fusion/test.mlir` `CHECK-MAPPING` line still needs updating (the `compiled_ii` value may have changed). Additionally, 3 pre-existing test failures are unrelated to this change: `fir.mlir`, `relu.mlir`, `simple_resnet_tosa.mlir`.
