import torch
import torch.nn as nn
from torch._inductor.decomposition import decompositions as inductor_decomp
import os


class SimpleResNetBlock(nn.Module):
    """
    Minimal ResNet Block: Conv -> ReLU -> Conv -> Add (residual)
    """

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + residual  # Residual connection
        out = torch.relu(out)
        return out


def generate_mlir():
    """Generate MLIR with Linalg ops"""
    model = SimpleResNetBlock(channels=64)
    model.eval()

    # Small input for quick testing: [batch, channels, height, width]
    x = torch.randn(1, 64, 8, 8)

    # Export to MLIR via torch-mlir
    try:
        from torch_mlir import compile

        mlir_module = compile(
            model, x, output_type="linalg-on-tensors", use_tracing=True
        )
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(output_dir, "Output")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "simple_resnet.mlir")
        with open(filename, "w") as f:
            f.write(str(mlir_module))
    except ImportError:
        print("Error: torch-mlir is not installed.\n")


if __name__ == "__main__":
    generate_mlir()
