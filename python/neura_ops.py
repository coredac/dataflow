"""Neura custom ops -- PyTorch frontend interface for hardware primitives.

Defines PyTorch custom ops that map to Neura dialect hardware primitives.
These ops are semantically equivalent to standard operations on the PyTorch
side (training and verification work as usual), but torch_mlir preserves
their op identity after tracing so that downstream compiler passes can
recognize and lower them to the corresponding Neura IR operations.

Supported custom ops:
  - neura::gather  ->  neura.gather (batched random-address read).

Usage:
  import neura_ops
  features = torch.ops.neura.gather(table, indices)
"""

import torch

# ============================================================================
#  neura::gather -- Batched indirect-address read.
#
#  Semantics:  table[indices]  (fancy indexing).
#  Hardware:   neura.gather -- Issues multiple random-address read requests
#              in a single cycle, exploiting memory-level parallelism for
#              hash-table lookups.
#
#  After torch_mlir tracing the op appears as
#      torch.operator "neura.gather"
#  and is lowered to neura.gather by LowerTorchCustomToNeuraPass.
# ============================================================================

# Registers the custom op schema (compatible with PyTorch 2.1+).
_neura_lib_def = torch.library.Library("neura", "DEF")
_neura_lib_def.define("gather(Tensor table, Tensor indices) -> Tensor")

# CPU implementation.
_neura_lib_impl = torch.library.Library("neura", "IMPL")


def _neura_gather_cpu(
    table: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Performs batched indirect-address read (hardware gather primitive).

    Args:
        table: Embedding table of shape [T, C].
        indices: Index vector of shape [K] with values in [0, T).

    Returns:
        Rows selected by ``indices``, shape [K, C].
    """
    return table[indices.long()]


_neura_lib_impl.impl("gather", _neura_gather_cpu)

# Meta / FakeTensor implementation (shape inference during torch_mlir tracing).
_neura_lib_meta = torch.library.Library("neura", "IMPL")


def _neura_gather_meta(
    table: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Returns an empty tensor with the correct output shape."""
    return torch.empty(
        (*indices.shape, *table.shape[1:]),
        dtype=table.dtype,
        device=table.device,
    )


try:
    _neura_lib_meta.impl("gather", _neura_gather_meta, "Meta")
except Exception:
    pass
