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

def generate_mlir(out_file,
                  src_len=48, tgt_len=32, d_model=64):
    if _try_torch_mlir(out_file, src_len, tgt_len, d_model):
        return
    print("Fail to generate MLIR via torch-mlir.")


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

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
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
        print(f"Generated {out_file}  [dynamic shapes via torch-mlir]")
        return True
    except Exception as e:
        print(f"Fail to generate MLIR via torch-mlir ({e})")
    return False


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "cross_attention_linalg.mlir"
    generate_mlir(out_file=out)
