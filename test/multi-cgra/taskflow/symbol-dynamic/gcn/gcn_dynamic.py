"""
Dense 2-Layer GCN — Multi-Task Pipeline with Dynamic N (number of nodes)

Real-world application:
    Graph Neural Networks with dense adjacency are used in:
    - Drug discovery: predicting molecular properties (each molecule is a graph
      with 20-1000 atoms as nodes)
    - Social network analysis: node classification on community subgraphs
    - Protein structure prediction: residue interaction graphs
    The number of nodes N varies per graph instance. A small molecule has ~20
    atoms; a protein has ~1000 residues.  The compiler cannot know N at compile
    time, but N is constant for a given input graph.

Multi-task pipeline (each becomes a separate taskflow.task):
    Task 0-1:  H0 = A @ X          [N, N] × [N, Fin]  → [N, Fin]   aggregate 1
    Task 2-3:  H1 = H0 @ W1        [N, Fin] × [Fin, Fh] → [N, Fh]  combine
    Task 4:    H2 = relu(H1)       [N, Fh] → [N, Fh]               activation
    Task 5-6:  H3 = A @ H2         [N, N] × [N, Fh]  → [N, Fh]     aggregate 2
    Task 7:    out = mean(H3, dim=0) [N, Fh] → [Fh]                 global pool
    → 8 tasks total, all with dynamic N bound

Dynamic dimension:
    N (number of nodes): symbol-dynamic — varies per graph in an inference batch.
"""

import torch
import torch.nn as nn


class DenseTwoLayerGCN(nn.Module):
    """2-layer GCN with dense adjacency: agg → combine → relu → agg → pool."""

    def __init__(self, in_features=8, hidden_features=16):
        super().__init__()
        self.combine = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x, adj):
        # x: [N, Fin],  adj: [N, N]  (normalized adjacency)
        h = torch.matmul(adj, x)          # aggregate 1: [N, Fin]
        h = self.combine(h)               # combine:     [N, Fh]
        h = torch.relu(h)                 # activation
        h = torch.matmul(adj, h)          # aggregate 2: [N, Fh]
        out = torch.mean(h, dim=0)        # global pool: [Fh]
        return out


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file="gcn_dynamic_linalg.mlir",
    num_nodes=32,
    in_features=8,
    hidden_features=16,
):
    if _try_torch_mlir(out_file, num_nodes, in_features, hidden_features):
        return
    _write_fallback_affine(out_file, in_features, hidden_features)


def _try_torch_mlir(out_file, N, Fin, Fh):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}), using fallback.")
        return False

    model = DenseTwoLayerGCN(in_features=Fin, hidden_features=Fh).eval()
    x = torch.randn(N, Fin)
    adj = torch.eye(N) / N

    for attempt in ("dynamic", "static"):
        try:
            kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
            if attempt == "dynamic":
                from torch.export import Dim
                n_dim = Dim("n", min=1, max=4096)
                kwargs["dynamic_shapes"] = {
                    "x": {0: n_dim},
                    "adj": {0: n_dim, 1: n_dim},
                }
            mlir_module = export_and_import(model, x, adj, **kwargs)
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


def _write_fallback_affine(out_file, Fin, Fh):
    NN = "memref<?x?xf32>"            # [N, N] adjacency
    NI = f"memref<?x{Fin}xf32>"       # [N, Fin]
    NH = f"memref<?x{Fh}xf32>"        # [N, Fh]
    IH = f"memref<{Fin}x{Fh}xf32>"    # [Fin, Fh] weight
    OH = f"memref<{Fh}xf32>"          # [Fh] pooled

    blocks = []
    blocks.append(f"""\
// ===----------------------------------------------------------------------===
// Dense 2-Layer GCN — affine MLIR with symbol-dynamic N (number of nodes)
//   8 top-level affine.for nests → 8 taskflow.task ops after conversion
// ===----------------------------------------------------------------------===
module {{
  func.func @forward(
      %X: {NI}, %adj: {NN}, %W1: {IH}) -> {OH} {{
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %N = memref.dim %X, %c0 : {NI}

    %alloc_agg1 = memref.alloc(%N) : {NI}
    %alloc_comb = memref.alloc(%N) : {NH}
    %alloc_relu = memref.alloc(%N) : {NH}
    %alloc_agg2 = memref.alloc(%N) : {NH}
    %alloc_pool = memref.alloc() : {OH}""")

    # Aggregate 1: H0 = adj @ X   [N,N] × [N,Fin] → [N,Fin]
    blocks.append(_mm("%alloc_agg1", "%adj", NN, "%X", NI, NI,
                       "%N", Fin, "%N",
                       "Task 0-1: H0 = adj @ X  (aggregate 1)"))

    # Combine: H1 = H0 @ W1   [N,Fin] × [Fin,Fh] → [N,Fh]
    blocks.append(_mm("%alloc_comb", "%alloc_agg1", NI, "%W1", IH, NH,
                       "%N", Fh, Fin,
                       "Task 2-3: H1 = H0 @ W1  (combine)"))

    # ReLU
    blocks.append(_relu("%alloc_relu", "%alloc_comb", NH, "%N", Fh,
                         "Task 4: H2 = relu(H1)"))

    # Aggregate 2: H3 = adj @ H2   [N,N] × [N,Fh] → [N,Fh]
    blocks.append(_mm("%alloc_agg2", "%adj", NN, "%alloc_relu", NH, NH,
                       "%N", Fh, "%N",
                       "Task 5-6: H3 = adj @ H2  (aggregate 2)"))

    # Global mean pool over N: out[j] = (1/N) * sum_i H3[i,j]
    blocks.append(f"""\
    // Task 7: out = mean(H3, dim=0)  — global pool over nodes
    affine.for %j = 0 to {Fh} {{
      affine.store %cst, %alloc_pool[%j] : {OH}
    }}
    affine.for %i = 0 to %N {{
      affine.for %j = 0 to {Fh} {{
        %0 = affine.load %alloc_agg2[%i, %j] : {NH}
        %1 = affine.load %alloc_pool[%j] : {OH}
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_pool[%j] : {OH}
      }}
    }}""")

    blocks.append(f"""\
    return %alloc_pool : {OH}
  }}
}}""")

    mlir = "\n\n".join(blocks)
    with open(out_file, "w") as f:
        f.write(mlir)
    print(f"Generated {out_file}  [fallback affine MLIR, dynamic N]")


if __name__ == "__main__":
    generate_mlir()
