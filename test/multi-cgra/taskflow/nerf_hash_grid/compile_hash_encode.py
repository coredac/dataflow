"""Compiles hash_encode to clean Torch Dialect MLIR with neura.gather preserved.

Uses a two-step approach to work around torch_mlir's backend contract
checker rejecting the custom neura.gather op:
  Step 1 -- Exports the model via OutputType.RAW (no backend lowering).
  Step 2 -- Runs cleanup passes manually (inline, DCE, calling conventions)
            while skipping the backend contract verifier.

The resulting MLIR retains ``torch.operator "neura.gather"`` and is
compact enough for downstream debugging and lowering.

Usage (inside Docker container):
    cd /workspace/dataflow/test/multi-cgra/taskflow/nerf_hash_grid
    python compile_hash_encode.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'python'))
import neura_ops  # noqa: E402

import torch
import torch.nn as nn
import torch_mlir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

from nerf_kernels import hash_encode


class HashEncodeFn(nn.Module):
    """Wrapper that passes embeddings as a function argument.

    Avoids registering embeddings as nn.Parameter so that torch_mlir's
    InlineGlobalSlots pass does not need to analyze the custom op.
    Offsets are stored as a plain Python list (compile-time constants).
    """

    def __init__(self, num_levels=2, level_dim=2, base_resolution=16,
                 per_level_scale=2.0, log2_hashmap_size=8, bound=1.0):
        super().__init__()
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.bound = bound

        # Precomputes offsets as a Python list (not a tensor buffer).
        import numpy as np
        S = np.log2(per_level_scale)
        max_params = 2 ** log2_hashmap_size
        offsets = [0]
        for level in range(num_levels):
            scale = np.exp2(level * S) * base_resolution - 1.0
            resolution = int(np.ceil(scale)) + 1
            n_dense = (resolution + 1) ** 3
            n_params = min(n_dense, max_params)
            offsets.append(offsets[-1] + n_params)
        self.offsets_list = offsets
        self.total_params = offsets[-1]

    def forward(self, inputs, embeddings):
        """Delegates to nerf_kernels.hash_encode.

        Args:
            inputs: Coordinates of shape [N, 3], float32, in [-bound, bound].
            embeddings: Embedding table of shape [total_params, C], float32.

        Returns:
            Encoded features of shape [N, num_levels * level_dim], float32.
        """
        offsets = torch.tensor(self.offsets_list, dtype=torch.int32)
        return hash_encode(
            inputs, embeddings, offsets, self.bound,
            num_levels=self.num_levels,
            level_dim=self.level_dim,
            base_resolution=self.base_resolution,
            per_level_scale=self.per_level_scale,
            log2_hashmap_size=self.log2_hashmap_size,
        )


# Cleanup passes to run on the RAW output.
# Skips the backend contract verifier that rejects neura.gather.
_CLEANUP_PIPELINE = (
    "builtin.module("
    "torch-prepare-for-globalize-object-graph,"
    "torch-globalize-object-graph,"
    "symbol-dce,"
    "inline,"
    "torch-adjust-calling-conventions"
    ")"
)


def compile_hash_encode(output_file="hash_encode_torch_clean.mlir"):
    """Compiles hash_encode and writes clean Torch Dialect MLIR to disk.

    Args:
        output_file: Path to the output MLIR file.

    Returns:
        The MLIR module string.
    """
    model = HashEncodeFn()
    model.eval()

    N = 4
    inputs = torch.randn(N, 3).clamp(-1.0, 1.0)
    embeddings = torch.randn(model.total_params, model.level_dim) * 0.01

    print(f"Model: num_levels={model.num_levels}, level_dim={model.level_dim}, "
          f"total_params={model.total_params}, offsets={model.offsets_list}")
    print(f"Inputs: inputs={inputs.shape}, embeddings={embeddings.shape}")

    # Step 1: Exports via RAW mode (bypasses backend lowering entirely).
    print("\nStep 1: Exporting RAW TorchScript MLIR...")
    raw_module = torch_mlir.compile(
        model, (inputs, embeddings),
        output_type=torch_mlir.OutputType.RAW,
        use_tracing=True,
    )
    raw_lines = str(raw_module).count("\n")
    print(f"  RAW output: {raw_lines} lines")

    # Step 2: Runs cleanup passes to reduce IR size.
    print("Step 2: Running cleanup passes...")
    run_pipeline_with_repro_report(
        raw_module,
        _CLEANUP_PIPELINE,
        "Cleanup passes (inline + DCE + calling conventions)",
    )

    mlir_str = str(raw_module)
    lines = mlir_str.count("\n")
    gather_count = mlir_str.count("neura.gather")

    print(f"  Clean output: {lines} lines")
    print(f"  neura.gather: {gather_count}")

    with open(output_file, "w") as f:
        f.write(mlir_str)
    print(f"\nSaved to: {output_file}")

    # Verification.
    aten_index = mlir_str.count("aten.index.Tensor")
    if gather_count > 0 and aten_index == 0:
        print("PASS: neura.gather preserved, aten.index.Tensor eliminated.")
    else:
        print(f"WARN: neura.gather={gather_count}, aten.index.Tensor={aten_index}")

    return mlir_str


if __name__ == "__main__":
    compile_hash_encode()
