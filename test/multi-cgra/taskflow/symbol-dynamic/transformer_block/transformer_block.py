"""
Transformer Encoder Block — Multi-Task Pipeline with Dynamic seq_len

Real-world application:
    Core building block of ALL large language models (GPT, BERT, LLaMA, T5),
    vision transformers (ViT, DeiT), and audio models (Whisper, HuBERT).
    seq_len is inherently dynamic: a short prompt "Hi" has seq_len=2 while a
    long document has seq_len=2048. In production LLM serving (vLLM,
    TensorRT-LLM), every request has a different sequence length.

Multi-task pipeline (each top-level affine.for becomes a separate taskflow.task):
    Task 0-1:  Q = X @ Wq           [S, D] × [D, D] → [S, D]   (init + matmul)
    Task 2-3:  K = X @ Wk           [S, D] × [D, D] → [S, D]   (init + matmul)
    Task 4-5:  V = X @ Wv           [S, D] × [D, D] → [S, D]   (init + matmul)
    Task 6-7:  scores = Q @ K^T     [S, D] × [D, S] → [S, S]   (init + matmul)
    Task 8:    attn = relu(scores)   [S, S] → [S, S]            (approx softmax)
    Task 9-10: ctx = attn @ V        [S, S] × [S, D] → [S, D]  (init + matmul)
    Task 11-12: out = ctx @ Wo       [S, D] × [D, D] → [S, D]  (init + matmul)
    Task 13:   res1 = X + out        [S, D]                     (residual add)
    Task 14-15: up = res1 @ W1       [S, D] × [D, FF] → [S, FF](init + matmul)
    Task 16:   act = relu(up)        [S, FF]                    (activation)
    Task 17-18: down = act @ W2      [S, FF] × [FF, D] → [S, D](init + matmul)
    Task 19:   res2 = res1 + down    [S, D]                     (residual add)
    → 20 tasks total, all with dynamic seq_len (S) bound

Dynamic dimension:
    S (seq_len): symbol-dynamic — unknown at compile time, constant per invocation.
    TaskflowCounterOp upper_bound set at runtime before each inference call.
    TaskDivisibilityAnalysis reports parallel_space = [-1] for all S-dependent loops.
"""

import math
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Single-head Transformer encoder: self-attention + FFN with residual."""

    def __init__(self, d_model=64, d_ff=256):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # x: [seq_len, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(0, 1)) * (1.0 / math.sqrt(self.d_model))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        attn_out = self.out_proj(context)
        x = x + attn_out
        ff_out = self.ff2(torch.relu(self.ff1(x)))
        x = x + ff_out
        return x


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file="transformer_block_linalg.mlir",
    seq_len=32,
    d_model=64,
    d_ff=256,
):
    """Export Transformer encoder block to MLIR."""
    if _try_torch_mlir(out_file, seq_len, d_model, d_ff):
        return
    _write_fallback_affine(out_file, d_model, d_ff)


def _try_torch_mlir(out_file, seq_len, d_model, d_ff):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}), using fallback.")
        return False

    model = TransformerEncoderBlock(d_model, d_ff).eval()
    x = torch.randn(seq_len, d_model)

    # Try dynamic shapes first, fall back to static.
    for attempt in ("dynamic", "static"):
        try:
            kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
            if attempt == "dynamic":
                from torch.export import Dim
                seq = Dim("seq", min=1, max=4096)
                kwargs["dynamic_shapes"] = {"x": {0: seq}}
            mlir_module = export_and_import(model, x, **kwargs)
            with open(out_file, "w") as f:
                f.write(str(mlir_module))
            tag = "dynamic" if attempt == "dynamic" else "static (fallback)"
            print(f"Generated {out_file}  [{tag} shapes via torch-mlir]")
            return True
        except Exception as e:
            if attempt == "dynamic":
                print(f"  dynamic_shapes failed ({e}), retrying static...")
            else:
                print(f"  static export also failed ({e})")
    return False


# ---------------------------------------------------------------------------
# Fallback: hand-written affine MLIR with symbol-dynamic seq_len
# ---------------------------------------------------------------------------

def _mm(alloc, A, A_ty, B, B_ty, C_ty, M, N, K, comment):
    """Matmul block: C = A @ B  (init zeros + triple-nested loop)."""
    return f"""\
    // {comment}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.store %cst, {alloc}[%i, %j] : {C_ty}
      }}
    }}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.for %k = 0 to {K} {{
          %0 = affine.load {A}[%i, %k] : {A_ty}
          %1 = affine.load {B}[%k, %j] : {B_ty}
          %2 = affine.load {alloc}[%i, %j] : {C_ty}
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, {alloc}[%i, %j] : {C_ty}
        }}
      }}
    }}"""


def _mm_transB(alloc, A, A_ty, B, B_ty, C_ty, M, N, K, comment):
    """Matmul C = A @ B^T  where B is [N, K], accessed as B[j, k]."""
    return f"""\
    // {comment}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.store %cst, {alloc}[%i, %j] : {C_ty}
      }}
    }}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.for %k = 0 to {K} {{
          %0 = affine.load {A}[%i, %k] : {A_ty}
          %1 = affine.load {B}[%j, %k] : {B_ty}
          %2 = affine.load {alloc}[%i, %j] : {C_ty}
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, {alloc}[%i, %j] : {C_ty}
        }}
      }}
    }}"""


def _relu(out, inp, inp_ty, out_ty, M, N, comment):
    return f"""\
    // {comment}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        %0 = affine.load {inp}[%i, %j] : {inp_ty}
        %1 = arith.maximumf %0, %cst : f32
        affine.store %1, {out}[%i, %j] : {out_ty}
      }}
    }}"""


def _add(out, A, B, ty, M, N, comment):
    return f"""\
    // {comment}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        %0 = affine.load {A}[%i, %j] : {ty}
        %1 = affine.load {B}[%i, %j] : {ty}
        %2 = arith.addf %0, %1 : f32
        affine.store %2, {out}[%i, %j] : {ty}
      }}
    }}"""


def _write_fallback_affine(out_file, d_model, d_ff):
    D = d_model
    FF = d_ff
    # Type shorthand.  '?' = symbol-dynamic seq_len.
    SD = f"memref<?x{D}xf32>"       # [S, D]
    SS = "memref<?x?xf32>"          # [S, S]
    SF = f"memref<?x{FF}xf32>"      # [S, FF]
    DD = f"memref<{D}x{D}xf32>"     # [D, D]  (weight)
    DF = f"memref<{D}x{FF}xf32>"    # [D, FF] (weight)
    FD = f"memref<{FF}x{D}xf32>"    # [FF, D] (weight)

    blocks = []
    blocks.append(f"""\
// ===----------------------------------------------------------------------===
// Transformer Encoder Block — affine MLIR with symbol-dynamic seq_len
//   20 top-level affine.for nests → 20 taskflow.task ops after conversion
//   All seq_len loops use %S from memref.dim (unknown at compile time)
// ===----------------------------------------------------------------------===
module {{
  func.func @forward(
      %X: {SD},
      %Wq: {DD}, %Wk: {DD}, %Wv: {DD}, %Wo: {DD},
      %W1: {DF}, %W2: {FD}) -> {SD} {{
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %S = memref.dim %X, %c0 : {SD}

    %alloc_q = memref.alloc(%S) : {SD}
    %alloc_k = memref.alloc(%S) : {SD}
    %alloc_v = memref.alloc(%S) : {SD}
    %alloc_scores = memref.alloc(%S, %S) : {SS}
    %alloc_attn = memref.alloc(%S, %S) : {SS}
    %alloc_ctx = memref.alloc(%S) : {SD}
    %alloc_out = memref.alloc(%S) : {SD}
    %alloc_res1 = memref.alloc(%S) : {SD}
    %alloc_up = memref.alloc(%S) : {SF}
    %alloc_act = memref.alloc(%S) : {SF}
    %alloc_down = memref.alloc(%S) : {SD}
    %alloc_res2 = memref.alloc(%S) : {SD}""")

    # Self-attention projections.
    blocks.append(_mm("%alloc_q", "%X", SD, "%Wq", DD, SD, "%S", D, D,
                       "Task 0-1: Q = X @ Wq"))
    blocks.append(_mm("%alloc_k", "%X", SD, "%Wk", DD, SD, "%S", D, D,
                       "Task 2-3: K = X @ Wk"))
    blocks.append(_mm("%alloc_v", "%X", SD, "%Wv", DD, SD, "%S", D, D,
                       "Task 4-5: V = X @ Wv"))

    # Attention scores: Q @ K^T  →  [S, S]
    blocks.append(_mm_transB("%alloc_scores", "%alloc_q", SD, "%alloc_k", SD,
                              SS, "%S", "%S", D,
                              "Task 6-7: scores = Q @ K^T (transposed matmul)"))

    # Approximate softmax as ReLU (exact softmax needs math.exp; torch-mlir
    # path produces correct softmax via TOSA).
    blocks.append(_relu("%alloc_attn", "%alloc_scores", SS, SS, "%S", "%S",
                         "Task 8: attn ≈ relu(scores)  [softmax in torch-mlir path]"))

    # Context: attn @ V  →  [S, D]
    blocks.append(_mm("%alloc_ctx", "%alloc_attn", SS, "%alloc_v", SD, SD,
                       "%S", D, "%S",
                       "Task 9-10: ctx = attn @ V"))

    # Output projection.
    blocks.append(_mm("%alloc_out", "%alloc_ctx", SD, "%Wo", DD, SD,
                       "%S", D, D,
                       "Task 11-12: out = ctx @ Wo"))

    # Residual add.
    blocks.append(_add("%alloc_res1", "%X", "%alloc_out", SD, "%S", D,
                        "Task 13: res1 = X + out (residual)"))

    # FFN.
    blocks.append(_mm("%alloc_up", "%alloc_res1", SD, "%W1", DF, SF,
                       "%S", FF, D,
                       "Task 14-15: up = res1 @ W1 (FFN up-projection)"))
    blocks.append(_relu("%alloc_act", "%alloc_up", SF, SF, "%S", FF,
                         "Task 16: act = relu(up)"))
    blocks.append(_mm("%alloc_down", "%alloc_act", SF, "%W2", FD, SD,
                       "%S", D, FF,
                       "Task 17-18: down = act @ W2 (FFN down-projection)"))
    blocks.append(_add("%alloc_res2", "%alloc_res1", "%alloc_down", SD, "%S", D,
                        "Task 19: res2 = res1 + down (residual)"))

    blocks.append(f"""\
    return %alloc_res2 : {SD}
  }}
}}""")

    mlir = "\n\n".join(blocks)
    with open(out_file, "w") as f:
        f.write(mlir)
    print(f"Generated {out_file}  [fallback affine MLIR, dynamic seq_len]")


if __name__ == "__main__":
    generate_mlir()
