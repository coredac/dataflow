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
    out_file,
    seq_len=32,
    d_model=64,
    d_ff=256,
):
    """Export Transformer encoder block to MLIR."""
    if _try_torch_mlir(out_file, seq_len, d_model, d_ff):
        return
    print("Fail to generate MLIR via torch-mlir.")

def _try_torch_mlir(out_file, seq_len, d_model, d_ff):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}).")
        return False

    model = TransformerEncoderBlock(d_model, d_ff).eval()
    x = torch.randn(seq_len, d_model)

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        seq = Dim("seq", min=1, max=4096)
        kwargs["dynamic_shapes"] = {"x": {0: seq}}
        mlir_module = export_and_import(model, x, **kwargs)
        with open(out_file, "w") as f:
            f.write(str(mlir_module))
        print(f"Generated {out_file}  [dynamic shapes via torch-mlir]")
        return True
    except Exception as e:
        print(f"Fail to generate MLIR via torch-mlir ({e})")
    return False


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "transformer_block_linalg.mlir"
    generate_mlir(out_file=out)
