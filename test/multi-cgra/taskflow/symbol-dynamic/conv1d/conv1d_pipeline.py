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
    out_file="conv1d_pipeline_linalg.mlir",
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
    _write_fallback_affine(out_file, in_channels, hidden_channels,
                            out_channels, num_classes, kernel_size)


def _try_torch_mlir(out_file, T, Cin, Ch, Cout, Ncls, K):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}), using fallback.")
        return False

    model = Conv1DPipeline(Cin, Ch, Cout, Ncls, K).eval()
    x = torch.randn(1, Cin, T)

    for attempt in ("dynamic", "static"):
        try:
            kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
            if attempt == "dynamic":
                from torch.export import Dim
                t_dim = Dim("time", min=2 * K, max=200000)
                kwargs["dynamic_shapes"] = {"x": {2: t_dim}}
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
# Fallback affine MLIR — 1D convolution as explicit loops
# ---------------------------------------------------------------------------

def _write_fallback_affine(out_file, Cin, Ch, Cout, Ncls, K):
    mlir = f"""\
// ===----------------------------------------------------------------------===
// Conv1D + Pool Pipeline — affine MLIR with symbol-dynamic time_len
//   9 top-level affine.for nests → 9 taskflow.task ops
//   T (time_len) is symbol-dynamic, derived from memref.dim
//   After conv1 (kernel={K}): output length = T - {K-1}
//   After conv2 (kernel={K}): output length = T - {2*(K-1)}
// ===----------------------------------------------------------------------===
#map_conv1_len = affine_map<()[s0] -> (s0 - {K - 1})>
#map_conv2_len = affine_map<()[s0] -> (s0 - {2 * (K - 1)})>

module {{
  func.func @forward(
      %X: memref<1x{Cin}x?xf32>,
      %F1: memref<{Ch}x{Cin}x{K}xf32>,
      %F2: memref<{Cout}x{Ch}x{K}xf32>,
      %Wfc: memref<{Cout}x{Ncls}xf32>) -> memref<1x{Ncls}xf32> {{
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %T = memref.dim %X, %c2 : memref<1x{Cin}x?xf32>
    %T1 = affine.apply #map_conv1_len()[%T]
    %T2 = affine.apply #map_conv2_len()[%T]

    // --- Allocations ---
    %alloc_c1 = memref.alloc(%T1) : memref<1x{Ch}x?xf32>
    %alloc_a1 = memref.alloc(%T1) : memref<1x{Ch}x?xf32>
    %alloc_c2 = memref.alloc(%T2) : memref<1x{Cout}x?xf32>
    %alloc_a2 = memref.alloc(%T2) : memref<1x{Cout}x?xf32>
    %alloc_pool = memref.alloc() : memref<1x{Cout}xf32>
    %alloc_y = memref.alloc() : memref<1x{Ncls}xf32>

    // Task 0: Init conv1 output
    affine.for %oc = 0 to {Ch} {{
      affine.for %t = 0 to %T1 {{
        affine.store %cst, %alloc_c1[%c0, %oc, %t] : memref<1x{Ch}x?xf32>
      }}
    }}

    // Task 1: Conv1D layer 1 — [1, {Cin}, T] * [{Ch}, {Cin}, {K}] → [1, {Ch}, T1]
    affine.for %oc = 0 to {Ch} {{
      affine.for %t = 0 to %T1 {{
        affine.for %ic = 0 to {Cin} {{
          affine.for %kk = 0 to {K} {{
            %0 = affine.load %X[%c0, %ic, %t + %kk] : memref<1x{Cin}x?xf32>
            %1 = affine.load %F1[%oc, %ic, %kk] : memref<{Ch}x{Cin}x{K}xf32>
            %2 = affine.load %alloc_c1[%c0, %oc, %t] : memref<1x{Ch}x?xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %alloc_c1[%c0, %oc, %t] : memref<1x{Ch}x?xf32>
          }}
        }}
      }}
    }}

    // Task 2: ReLU
    affine.for %oc = 0 to {Ch} {{
      affine.for %t = 0 to %T1 {{
        %0 = affine.load %alloc_c1[%c0, %oc, %t] : memref<1x{Ch}x?xf32>
        %1 = arith.maximumf %0, %cst : f32
        affine.store %1, %alloc_a1[%c0, %oc, %t] : memref<1x{Ch}x?xf32>
      }}
    }}

    // Task 3: Init conv2 output
    affine.for %oc = 0 to {Cout} {{
      affine.for %t = 0 to %T2 {{
        affine.store %cst, %alloc_c2[%c0, %oc, %t] : memref<1x{Cout}x?xf32>
      }}
    }}

    // Task 4: Conv1D layer 2 — [1, {Ch}, T1] * [{Cout}, {Ch}, {K}] → [1, {Cout}, T2]
    affine.for %oc = 0 to {Cout} {{
      affine.for %t = 0 to %T2 {{
        affine.for %ic = 0 to {Ch} {{
          affine.for %kk = 0 to {K} {{
            %0 = affine.load %alloc_a1[%c0, %ic, %t + %kk] : memref<1x{Ch}x?xf32>
            %1 = affine.load %F2[%oc, %ic, %kk] : memref<{Cout}x{Ch}x{K}xf32>
            %2 = affine.load %alloc_c2[%c0, %oc, %t] : memref<1x{Cout}x?xf32>
            %3 = arith.mulf %0, %1 : f32
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %alloc_c2[%c0, %oc, %t] : memref<1x{Cout}x?xf32>
          }}
        }}
      }}
    }}

    // Task 5: ReLU
    affine.for %oc = 0 to {Cout} {{
      affine.for %t = 0 to %T2 {{
        %0 = affine.load %alloc_c2[%c0, %oc, %t] : memref<1x{Cout}x?xf32>
        %1 = arith.maximumf %0, %cst : f32
        affine.store %1, %alloc_a2[%c0, %oc, %t] : memref<1x{Cout}x?xf32>
      }}
    }}

    // Task 6: Global average pool over time — [1, Cout, T2] → [1, Cout]
    affine.for %oc = 0 to {Cout} {{
      affine.store %cst, %alloc_pool[%c0, %oc] : memref<1x{Cout}xf32>
    }}
    affine.for %oc = 0 to {Cout} {{
      affine.for %t = 0 to %T2 {{
        %0 = affine.load %alloc_a2[%c0, %oc, %t] : memref<1x{Cout}x?xf32>
        %1 = affine.load %alloc_pool[%c0, %oc] : memref<1x{Cout}xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_pool[%c0, %oc] : memref<1x{Cout}xf32>
      }}
    }}

    // Task 7: Init FC output
    affine.for %j = 0 to {Ncls} {{
      affine.store %cst, %alloc_y[%c0, %j] : memref<1x{Ncls}xf32>
    }}

    // Task 8: FC — [1, Cout] × [Cout, Ncls] → [1, Ncls]
    affine.for %j = 0 to {Ncls} {{
      affine.for %k = 0 to {Cout} {{
        %0 = affine.load %alloc_pool[%c0, %k] : memref<1x{Cout}xf32>
        %1 = affine.load %Wfc[%k, %j] : memref<{Cout}x{Ncls}xf32>
        %2 = affine.load %alloc_y[%c0, %j] : memref<1x{Ncls}xf32>
        %3 = arith.mulf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %alloc_y[%c0, %j] : memref<1x{Ncls}xf32>
      }}
    }}

    return %alloc_y : memref<1x{Ncls}xf32>
  }}
}}
"""
    with open(out_file, "w") as f:
        f.write(mlir)
    print(f"Generated {out_file}  [fallback affine MLIR, dynamic time_len]")


if __name__ == "__main__":
    generate_mlir()
