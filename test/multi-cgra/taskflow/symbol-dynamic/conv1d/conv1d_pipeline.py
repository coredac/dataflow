"""
Conv1D + Pooling Pipeline — Multi-Task Pipeline with Dynamic time_len

Real-world application:
    - Audio classification (speech command recognition, environmental sound)
    - ECG / biomedical signal analysis (heartbeat anomaly detection)
    - Time-series forecasting (sensor data, financial data)
    Audio clips have variable length: a 1-second clip at 16kHz has 16000
    samples; a 10-second clip has 160000.  ECG segments vary by patient and
    recording duration.  The model is compiled once; runtime adapts to the
    actual signal length.

Multi-task pipeline:
    Task 0-1:  C1 = conv1d(X, F1)     [T, Cin] × [K, Cin, Ch] → [T-K+1, Ch]
    Task 2:    A1 = relu(C1)          [T', Ch]
    Task 3-4:  C2 = conv1d(A1, F2)    [T', Ch] × [K, Ch, Cout] → [T'', Cout]
    Task 5:    A2 = relu(C2)          [T'', Cout]
    Task 6:    P = global_avg_pool(A2) [T'', Cout] → [Cout]
    Task 7-8:  Y = P @ Wfc            [Cout] × [Cout, Ncls] → [Ncls]
    → 9 tasks total, with dynamic time dimension T

    Note: After each conv, the time dimension shrinks by (kernel_size - 1).
    All time-dependent loops have symbol-dynamic bounds derived from T.

Dynamic dimension:
    T (time_len): symbol-dynamic — varies per input signal.
"""

import torch
import torch.nn as nn


class Conv1DPipeline(nn.Module):
    """Conv1D → ReLU → Conv1D → ReLU → GlobalAvgPool → Linear."""

    def __init__(self, in_channels=1, hidden_channels=16,
                 out_channels=32, num_classes=10, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels,
                               kernel_size, bias=False)
        self.conv2 = nn.Conv1d(hidden_channels, out_channels,
                               kernel_size, bias=False)
        self.fc = nn.Linear(out_channels, num_classes, bias=False)

    def forward(self, x):
        # x: [1, in_channels, time_len]  (batch=1)
        h = torch.relu(self.conv1(x))       # [1, Ch, T-K+1]
        h = torch.relu(self.conv2(h))       # [1, Cout, T-2K+2]
        h = torch.mean(h, dim=2)            # [1, Cout]  global avg pool
        return self.fc(h)                    # [1, Ncls]


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file,
    time_len=128,
    in_channels=1,
    hidden_channels=16,
    out_channels=32,
    num_classes=10,
    kernel_size=3,
):
    if _try_torch_mlir(out_file, time_len, in_channels, hidden_channels,
                        out_channels, num_classes, kernel_size):
        return
    print("Fail to generate MLIR via torch-mlir.")


def _try_torch_mlir(out_file, T, Cin, Ch, Cout, Ncls, K):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}), using fallback.")
        return False

    model = Conv1DPipeline(Cin, Ch, Cout, Ncls, K).eval()
    x = torch.randn(1, Cin, T)

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        t_dim = Dim("time", min=2 * K, max=200000)
        kwargs["dynamic_shapes"] = {"x": {2: t_dim}}
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
    out = sys.argv[1] if len(sys.argv) > 1 else "conv1d_pipeline_linalg.mlir"
    generate_mlir(out_file=out)
