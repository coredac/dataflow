"""
Encoder-Decoder Cross-Attention — Multi-Task Pipeline with Two Dynamic Dims

Real-world application:
    - Machine translation (T5, mBART, NLLB):  source and target sentences
      have *different* and *variable* lengths — "Hello" (5 tokens) translated
      to "Bonjour" (7 tokens).
    - Speech recognition / Whisper:  audio frames (src_len ≈ 1500 for 30 s)
      are attended by generated text tokens (tgt_len varies per utterance).
    - Image captioning (encoder=ViT patches, decoder=caption tokens).

    This benchmark is distinct from self-attention (#1) because Q comes from
    the *target* sequence while K, V come from the *source* sequence.  The
    score matrix is [tgt_len × src_len], mixing two independent dynamic dims.

Multi-task pipeline:
    Task 0-1:   Q  = Tgt @ Wq        [Tt, D] × [D, D]  → [Tt, D]
    Task 2-3:   K  = Src @ Wk        [Ts, D] × [D, D]  → [Ts, D]
    Task 4-5:   V  = Src @ Wv        [Ts, D] × [D, D]  → [Ts, D]
    Task 6-7:   S  = Q @ K^T         [Tt, D] × [D, Ts] → [Tt, Ts]
    Task 8:     A  = softmax(S)      [Tt, Ts]           (omitted in fallback)
    Task 9-10:  Ctx = A @ V          [Tt, Ts] × [Ts, D] → [Tt, D]
    Task 11-12: Out = Ctx @ Wo       [Tt, D] × [D, D]  → [Tt, D]
    → 13 tasks total, src_len and tgt_len are independent dynamic dims.

Dynamic dimensions:
    src_len (Ts): varies per source sentence / audio clip.
    tgt_len (Tt): varies per target sentence / generated output.
    Both are symbol-dynamic — determined once before running the pipeline.
"""

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """Encoder-decoder cross-attention: Q from target, K/V from source."""

    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, tgt, src):
        # tgt: [tgt_len, d_model],  src: [src_len, d_model]
        q = self.q_proj(tgt)                                          # [Tt, D]
        k = self.k_proj(src)                                          # [Ts, D]
        v = self.v_proj(src)                                          # [Ts, D]

        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.matmul(q, k.transpose(0, 1)) * scale          # [Tt, Ts]
        attn = torch.softmax(scores, dim=-1)                          # [Tt, Ts]
        context = torch.matmul(attn, v)                               # [Tt, D]

        return self.out_proj(context)                                 # [Tt, D]


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(out_file="cross_attention_linalg.mlir",
                  src_len=48, tgt_len=32, d_model=64):
    if _try_torch_mlir(out_file, src_len, tgt_len, d_model):
        return
    _write_fallback_affine(out_file, d_model)


def _try_torch_mlir(out_file, Ts, Tt, D):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}), using fallback.")
        return False

    model = CrossAttention(D).eval()
    tgt = torch.randn(Tt, D)
    src = torch.randn(Ts, D)

    for attempt in ("dynamic", "static"):
        try:
            kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
            if attempt == "dynamic":
                from torch.export import Dim
                ts_dim = Dim("src_len", min=1, max=4096)
                tt_dim = Dim("tgt_len", min=1, max=4096)
                kwargs["dynamic_shapes"] = {
                    "tgt": {0: tt_dim},
                    "src": {0: ts_dim},
                }
            mlir_module = export_and_import(model, tgt, src, **kwargs)
            with open(out_file, "w") as f:
                f.write(str(mlir_module))
            tag = "dynamic" if attempt == "dynamic" else "static"
            print(f"Generated {out_file}  [{tag} shapes via torch-mlir]")
            return True
        except Exception as e:
            if attempt == "dynamic":
                print(f"  dynamic_shapes failed ({e}), retrying static...")
            else:
                print(f"  static export also failed ({e})")
    return False


# ---------------------------------------------------------------------------
# Helpers for fallback affine MLIR generation
# ---------------------------------------------------------------------------

def _mm(C, A, A_ty, B, B_ty, C_ty, M, N, K, comment, alloc_args):
    """Return MLIR lines for C = A @ B  (init + matmul)."""
    a = alloc_args  # e.g. "%Ts, %Tt" or "%Ts" or ""
    return f"""\
    // {comment}  —  init
    {C} = memref.alloc({a}) : {C_ty}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.store %cst, {C}[%i, %j] : {C_ty}
      }}
    }}
    // {comment}  —  compute
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.for %k = 0 to {K} {{
          %0 = affine.load {A}[%i, %k] : {A_ty}
          %1 = affine.load {B}[%k, %j] : {B_ty}
          %2 = affine.load {C}[%i, %j] : {C_ty}
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, {C}[%i, %j] : {C_ty}
        }}
      }}
    }}"""


def _mm_transB(C, A, A_ty, B, B_ty, C_ty, M, N, K, comment, alloc_args):
    """Return MLIR lines for C = A @ B^T  (B is [N,K], accessed [j,k])."""
    a = alloc_args
    return f"""\
    // {comment}  —  init
    {C} = memref.alloc({a}) : {C_ty}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.store %cst, {C}[%i, %j] : {C_ty}
      }}
    }}
    // {comment}  —  compute  (B transposed)
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        affine.for %k = 0 to {K} {{
          %0 = affine.load {A}[%i, %k] : {A_ty}
          %1 = affine.load {B}[%j, %k] : {B_ty}
          %2 = affine.load {C}[%i, %j] : {C_ty}
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, {C}[%i, %j] : {C_ty}
        }}
      }}
    }}"""


# ---------------------------------------------------------------------------
# Fallback affine MLIR — cross-attention with two dynamic dims
# ---------------------------------------------------------------------------

def _write_fallback_affine(out_file, D):
    body_parts = []

    # Q = Tgt @ Wq  [Tt, D] x [D, D] → [Tt, D]
    body_parts.append(_mm(
        "%Q", "%tgt", f"memref<?x{D}xf32>",
        "%Wq", f"memref<{D}x{D}xf32>",
        f"memref<?x{D}xf32>", "%Tt", D, D,
        "Q = Tgt @ Wq", "%Tt"))

    # K = Src @ Wk  [Ts, D] x [D, D] → [Ts, D]
    body_parts.append(_mm(
        "%K", "%src", f"memref<?x{D}xf32>",
        "%Wk", f"memref<{D}x{D}xf32>",
        f"memref<?x{D}xf32>", "%Ts", D, D,
        "K = Src @ Wk", "%Ts"))

    # V = Src @ Wv  [Ts, D] x [D, D] → [Ts, D]
    body_parts.append(_mm(
        "%V", "%src", f"memref<?x{D}xf32>",
        "%Wv", f"memref<{D}x{D}xf32>",
        f"memref<?x{D}xf32>", "%Ts", D, D,
        "V = Src @ Wv", "%Ts"))

    # scores = Q @ K^T  [Tt, D] x [Ts, D]^T → [Tt, Ts]
    body_parts.append(_mm_transB(
        "%scores", "%Q", f"memref<?x{D}xf32>",
        "%K", f"memref<?x{D}xf32>",
        "memref<?x?xf32>", "%Tt", "%Ts", D,
        "scores = Q @ K^T", "%Tt, %Ts"))

    # NOTE: Softmax omitted — in torch-mlir output it expands to
    # max-reduce + exp + sum-reduce + div, adding ~4 separate tasks.
    # For fallback we pass scores through directly.

    # ctx = scores @ V  [Tt, Ts] x [Ts, D] → [Tt, D]
    body_parts.append(_mm(
        "%ctx", "%scores", "memref<?x?xf32>",
        "%V", f"memref<?x{D}xf32>",
        f"memref<?x{D}xf32>", "%Tt", D, "%Ts",
        "ctx = scores @ V", "%Tt"))

    # out = ctx @ Wo  [Tt, D] x [D, D] → [Tt, D]
    body_parts.append(_mm(
        "%out", "%ctx", f"memref<?x{D}xf32>",
        "%Wo", f"memref<{D}x{D}xf32>",
        f"memref<?x{D}xf32>", "%Tt", D, D,
        "Out = ctx @ Wo", "%Tt"))

    body = "\n\n".join(body_parts)

    mlir = f"""\
// ===----------------------------------------------------------------------===
// Cross-Attention — affine MLIR with TWO independent dynamic dims
//   Ts = src_len (source sequence),  Tt = tgt_len (target sequence)
//   13 top-level affine.for nests → 13 taskflow.task ops
//   Softmax omitted in fallback (torch-mlir path includes it).
// ===----------------------------------------------------------------------===

module {{
  func.func @forward(
      %tgt: memref<?x{D}xf32>,
      %src: memref<?x{D}xf32>,
      %Wq:  memref<{D}x{D}xf32>,
      %Wk:  memref<{D}x{D}xf32>,
      %Wv:  memref<{D}x{D}xf32>,
      %Wo:  memref<{D}x{D}xf32>) -> memref<?x{D}xf32> {{
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %Ts = memref.dim %src, %c0 : memref<?x{D}xf32>
    %Tt = memref.dim %tgt, %c0 : memref<?x{D}xf32>

{body}

    return %out : memref<?x{D}xf32>
  }}
}}
"""
    with open(out_file, "w") as f:
        f.write(mlir)
    print(f"Generated {out_file}  [fallback affine MLIR, dynamic src_len + tgt_len]")


if __name__ == "__main__":
    generate_mlir()
