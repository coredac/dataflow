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
    out_file,
    num_nodes=32,
    in_features=8,
    hidden_features=16,
):
    if _try_torch_mlir(out_file, num_nodes, in_features, hidden_features):
        return
    print("Fail to generate MLIR via torch-mlir.")


def _try_torch_mlir(out_file, N, Fin, Fh):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}).")
        return False

    model = DenseTwoLayerGCN(in_features=Fin, hidden_features=Fh).eval()
    x = torch.randn(N, Fin)
    adj = torch.eye(N) / N

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        n_dim = Dim("n", min=1, max=4096)
        kwargs["dynamic_shapes"] = {
            "x": {0: n_dim},
            "adj": {0: n_dim, 1: n_dim},
        }
        mlir_module = export_and_import(model, x, adj, **kwargs)
        with open(out_file, "w") as f:
            f.write(str(mlir_module))
        print(f"Generated {out_file}  [dynamic shapes via torch-mlir]")
        return True
    except Exception as e:
        print(f"Fail to generate MLIR via torch-mlir ({e})")
    return False

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "gcn_dynamic_linalg.mlir"
    generate_mlir(out_file=out)
