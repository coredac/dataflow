"""
3-Layer MLP Pipeline — Multi-Task Pipeline with Dynamic batch_size

Real-world application:
    Core compute of recommendation systems (Meta DLRM, Google Wide&Deep),
    tabular data inference, and the dense sub-networks inside larger models.
    In production serving, the batch size varies by traffic: a single user
    click triggers batch=1, while a bulk scoring job uses batch=256.
    The server compiles ONE configuration and the runtime adapts duplication
    based on actual batch size.

Multi-task pipeline:
    Task 0-1:  H1 = X @ W1          [B, D_in] × [D_in, D_h] → [B, D_h]
    Task 2:    A1 = relu(H1)        [B, D_h]
    Task 3-4:  H2 = A1 @ W2         [B, D_h] × [D_h, D_h] → [B, D_h]
    Task 5:    A2 = relu(H2)        [B, D_h]
    Task 6-7:  Y = A2 @ W3          [B, D_h] × [D_h, D_out] → [B, D_out]
    → 8 tasks total, all with dynamic batch dimension B

Dynamic dimension:
    B (batch_size): symbol-dynamic — varies per inference request.
"""

import torch
import torch.nn as nn


class ThreeLayerMLP(nn.Module):
    """3-layer MLP: linear → relu → linear → relu → linear."""

    def __init__(self, d_in=64, d_hidden=128, d_out=32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_hidden, bias=False)
        self.fc3 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        # x: [batch_size, d_in]
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file="mlp_pipeline_linalg.mlir",
    batch_size=32,
    d_in=64,
    d_hidden=128,
    d_out=32,
):
    if _try_torch_mlir(out_file, batch_size, d_in, d_hidden, d_out):
        return
    _write_fallback_affine(out_file, d_in, d_hidden, d_out)


def _try_torch_mlir(out_file, B, d_in, d_hidden, d_out):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}), using fallback.")
        return False

    model = ThreeLayerMLP(d_in, d_hidden, d_out).eval()
    x = torch.randn(B, d_in)

    for attempt in ("dynamic", "static"):
        try:
            kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
            if attempt == "dynamic":
                from torch.export import Dim
                batch = Dim("batch", min=1, max=4096)
                kwargs["dynamic_shapes"] = {"x": {0: batch}}
            mlir_module = export_and_import(model, x, **kwargs)
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
# Fallback affine MLIR
# ---------------------------------------------------------------------------

def _mm(alloc, A, A_ty, B, B_ty, C_ty, M, N, K, comment):
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


def _relu(out, inp, ty, M, N, comment):
    return f"""\
    // {comment}
    affine.for %i = 0 to {M} {{
      affine.for %j = 0 to {N} {{
        %0 = affine.load {inp}[%i, %j] : {ty}
        %1 = arith.maximumf %0, %cst : f32
        affine.store %1, {out}[%i, %j] : {ty}
      }}
    }}"""


def _write_fallback_affine(out_file, d_in, d_hidden, d_out):
    DI = d_in; DH = d_hidden; DO = d_out

    BI = f"memref<?x{DI}xf32>"
    BH = f"memref<?x{DH}xf32>"
    BO = f"memref<?x{DO}xf32>"
    WIH = f"memref<{DI}x{DH}xf32>"
    WHH = f"memref<{DH}x{DH}xf32>"
    WHO = f"memref<{DH}x{DO}xf32>"

    blocks = []
    blocks.append(f"""\
// ===----------------------------------------------------------------------===
// 3-Layer MLP Pipeline — affine MLIR with symbol-dynamic batch_size
//   8 top-level affine.for nests → 8 taskflow.task ops
// ===----------------------------------------------------------------------===
module {{
  func.func @forward(
      %X: {BI}, %W1: {WIH}, %W2: {WHH}, %W3: {WHO}) -> {BO} {{
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %B = memref.dim %X, %c0 : {BI}

    %alloc_h1 = memref.alloc(%B) : {BH}
    %alloc_a1 = memref.alloc(%B) : {BH}
    %alloc_h2 = memref.alloc(%B) : {BH}
    %alloc_a2 = memref.alloc(%B) : {BH}
    %alloc_y  = memref.alloc(%B) : {BO}""")

    blocks.append(_mm("%alloc_h1", "%X", BI, "%W1", WIH, BH,
                       "%B", DH, DI, "Task 0-1: H1 = X @ W1"))
    blocks.append(_relu("%alloc_a1", "%alloc_h1", BH, "%B", DH,
                         "Task 2: A1 = relu(H1)"))
    blocks.append(_mm("%alloc_h2", "%alloc_a1", BH, "%W2", WHH, BH,
                       "%B", DH, DH, "Task 3-4: H2 = A1 @ W2"))
    blocks.append(_relu("%alloc_a2", "%alloc_h2", BH, "%B", DH,
                         "Task 5: A2 = relu(H2)"))
    blocks.append(_mm("%alloc_y", "%alloc_a2", BH, "%W3", WHO, BO,
                       "%B", DO, DH, "Task 6-7: Y = A2 @ W3"))

    blocks.append(f"""\
    return %alloc_y : {BO}
  }}
}}""")

    mlir = "\n\n".join(blocks)
    with open(out_file, "w") as f:
        f.write(mlir)
    print(f"Generated {out_file}  [fallback affine MLIR, dynamic batch_size]")


if __name__ == "__main__":
    generate_mlir()
