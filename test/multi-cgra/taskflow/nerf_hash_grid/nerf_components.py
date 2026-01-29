"""NeRF components for modular MLIR compilation.

This module contains PyTorch implementations of NeRF components:
  - RaySampler: Samples 3D positions along rays
  - HashGridEncoder: Multi-resolution hash encoding (Instant-NGP style)
  - NeRFMLP: Neural network for density and color prediction
  - HashGridNeRF: Complete NeRF pipeline

These components are designed to be compiled individually to MLIR and then
combined into a modular heterogeneous computing system.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RaySampler(nn.Module):
  """Samples 3D positions along rays for volume rendering."""

  def __init__(self, num_samples=64, near=2.0, far=6.0):
    """Initializes ray sampler.

    Args:
      num_samples: Number of samples per ray.
      near: Near plane distance.
      far: Far plane distance.
    """
    super().__init__()
    self.num_samples = num_samples
    # Register constants as buffers to avoid torch.constant issues.
    self.register_buffer('near', torch.tensor(near, dtype=torch.float32))
    self.register_buffer('far', torch.tensor(far, dtype=torch.float32))

  def forward(self, rays_o, rays_d):
    """Samples positions along rays.

    Args:
      rays_o: Ray origins [batch_size, 3].
      rays_d: Ray directions [batch_size, 3].

    Returns:
      Sampled 3D positions [batch_size, num_samples, 3].
    """
    batch_size = rays_o.shape[0]

    # Manually implement linspace for compatibility.
    # Original: t_vals = torch.linspace(self.near, self.far, ...)
    # Compatible: Use arange + scaling.
    indices = torch.arange(
        self.num_samples, device=rays_o.device, dtype=rays_o.dtype)
    step = (self.far - self.near) / (self.num_samples - 1)
    t_vals = self.near + indices * step  # [num_samples]

    t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)  # [B, N]

    # positions = rays_o + t * rays_d
    positions = (rays_o.unsqueeze(1) +
                 t_vals.unsqueeze(2) * rays_d.unsqueeze(1))

    return positions  # [batch_size, num_samples, 3]


class HashGridEncoder(nn.Module):
  """Multi-resolution hash grid encoding (Instant-NGP style)."""

  def __init__(self,
               num_levels=16,
               features_per_level=2,
               log2_hashmap_size=19,
               base_resolution=16,
               finest_resolution=512):
    """Initializes hash grid encoder.

    Args:
      num_levels: Number of resolution levels.
      features_per_level: Feature dimension per level.
      log2_hashmap_size: Log2 of hash table size.
      base_resolution: Coarsest grid resolution.
      finest_resolution: Finest grid resolution.
    """
    super().__init__()
    self.num_levels = num_levels
    self.features_per_level = features_per_level
    self.log2_hashmap_size = log2_hashmap_size
    self.hashmap_size = 2**log2_hashmap_size
    self.base_resolution = base_resolution
    self.finest_resolution = finest_resolution

    # Compute resolution growth factor per level.
    self.b = np.exp(
        (np.log(finest_resolution) - np.log(base_resolution)) /
        (num_levels - 1))

    # Hash tables for each level (learnable parameters).
    self.hash_tables = nn.ParameterList([
        nn.Parameter(
            torch.randn(self.hashmap_size, features_per_level) * 0.01)
        for _ in range(num_levels)
    ])

  def hash_function(self, coords, level):
    """Hashes 3D integer coordinates to hash table indices.

    Uses modulo operation instead of bitwise operations for compatibility.
    Converts to int32 for better compatibility with downstream operations.

    Args:
      coords: Integer coordinates [batch_size, 3].
      level: Resolution level index.

    Returns:
      Hash indices [batch_size].
    """
    # Convert to int32 for compatibility.
    x = coords[:, 0].int()
    y = coords[:, 1].int()
    z = coords[:, 2].int()

    # Spatial hash using prime numbers (avoid int32 overflow).
    hashed = x * 1 + y * 73856093 + z * 19349663

    # Use modulo instead of bitwise AND.
    return hashed % self.hashmap_size

  def grid_sample_3d(self, positions, level):
    """Samples features from hash grid at given level using trilinear
    interpolation.

    Args:
      positions: Normalized positions [batch_size, num_samples, 3] in [0, 1].
      level: Resolution level index.

    Returns:
      Interpolated features [batch_size, num_samples, features_per_level].
    """
    batch_size, num_samples, _ = positions.shape
    resolution = int(np.floor(self.base_resolution * (self.b**level)))

    # Scale positions to grid resolution.
    scaled_pos = positions * (resolution - 1)  # [B, N, 3]

    # Get integer grid coordinates (8 corners of cube).
    base_coords = torch.floor(scaled_pos).int()  # [B, N, 3] - int32

    # Trilinear interpolation weights.
    frac = scaled_pos - base_coords.float()  # [B, N, 3]

    # Flatten batch and samples for processing.
    base_coords_flat = base_coords.view(-1, 3)  # [B*N, 3]
    frac_flat = frac.view(-1, 3)  # [B*N, 3]

    # Sample from 8 corners and compute trilinear interpolation.
    features_list = []
    for dx in [0, 1]:
      for dy in [0, 1]:
        for dz in [0, 1]:
          # Compute offset coordinates.
          offset_x = base_coords_flat[:, 0] + dx
          offset_y = base_coords_flat[:, 1] + dy
          offset_z = base_coords_flat[:, 2] + dz

          # Stack into coordinates.
          corner_coords = torch.stack([offset_x, offset_y, offset_z], dim=1)

          # Hash coordinates to table indices.
          indices = self.hash_function(corner_coords, level)  # [B*N]

          # Convert to long for tensor indexing.
          indices = indices.long()

          # Lookup features from hash table.
          corner_features = self.hash_tables[level][indices]  # [B*N, F]

          # Compute trilinear weight.
          weight = 1.0
          weight *= (1 - frac_flat[:, 0]) if dx == 0 else frac_flat[:, 0]
          weight *= (1 - frac_flat[:, 1]) if dy == 0 else frac_flat[:, 1]
          weight *= (1 - frac_flat[:, 2]) if dz == 0 else frac_flat[:, 2]

          features_list.append(corner_features * weight.unsqueeze(1))

    # Sum contributions from all corners.
    interpolated_features = sum(features_list)  # [B*N, F]

    # Reshape back.
    interpolated_features = interpolated_features.view(
        batch_size, num_samples, self.features_per_level)

    return interpolated_features

  def forward(self, positions):
    """Encodes 3D positions with multi-resolution hash encoding.

    Args:
      positions: 3D positions [batch_size, num_samples, 3] in range [-1, 1].

    Returns:
      Encoded features [batch_size, num_samples, num_levels *
      features_per_level].
    """
    # Normalize positions to [0, 1].
    positions_normalized = (positions + 1.0) / 2.0

    # Encode at all levels.
    encoded_features = []
    for level in range(self.num_levels):
      level_features = self.grid_sample_3d(positions_normalized, level)
      encoded_features.append(level_features)

    # Concatenate features from all levels.
    encoded = torch.cat(encoded_features, dim=-1)  # [B, N, L*F]

    return encoded


class NeRFMLP(nn.Module):
  """MLP for NeRF: predicts density and color from encoded features."""

  def __init__(self, input_dim=32, hidden_dim=64, num_layers=3):
    """Initializes NeRF MLP.

    Args:
      input_dim: Input feature dimension.
      hidden_dim: Hidden layer dimension.
      num_layers: Number of hidden layers.
    """
    super().__init__()

    # Density network.
    self.density_net = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
    for _ in range(num_layers - 1):
      self.density_net.append(nn.Linear(hidden_dim, hidden_dim))
    self.density_out = nn.Linear(hidden_dim, 1)

    # Color network (conditioned on view direction).
    self.color_net = nn.ModuleList(
        [nn.Linear(hidden_dim + 3, hidden_dim)])  # +3 for view direction
    for _ in range(num_layers - 2):
      self.color_net.append(nn.Linear(hidden_dim, hidden_dim))
    self.color_out = nn.Linear(hidden_dim, 3)

  def forward(self, encoded_features, view_dirs):
    """Predicts density and color from encoded features.

    Args:
      encoded_features: Encoded position features [batch_size, num_samples,
        input_dim].
      view_dirs: View directions [batch_size, 3].

    Returns:
      Tuple of:
        density: Volume density [batch_size, num_samples, 1].
        rgb: RGB color [batch_size, num_samples, 3].
    """
    batch_size, num_samples, _ = encoded_features.shape

    # Density prediction.
    x = encoded_features
    for layer in self.density_net:
      x = torch.relu(layer(x))
    density = self.density_out(x)  # [B, N, 1]

    # Get features for color prediction.
    density_features = x  # [B, N, hidden_dim]

    # Expand view directions.
    view_dirs_expanded = view_dirs.unsqueeze(1).expand(
        -1, num_samples, -1)  # [B, N, 3]

    # Concatenate density features with view directions.
    color_input = torch.cat([density_features, view_dirs_expanded], dim=-1)

    # Color prediction.
    x = color_input
    for layer in self.color_net:
      x = torch.relu(layer(x))
    rgb = torch.sigmoid(self.color_out(x))  # [B, N, 3]

    return density, rgb


class HashGridNeRF(nn.Module):
  """Complete NeRF pipeline with hash grid encoding."""

  def __init__(self,
               num_samples=64,
               num_levels=16,
               features_per_level=2,
               hidden_dim=64):
    """Initializes complete NeRF model.

    Args:
      num_samples: Number of samples per ray.
      num_levels: Number of hash grid levels.
      features_per_level: Features per hash grid level.
      hidden_dim: MLP hidden dimension.
    """
    super().__init__()
    self.ray_sampler = RaySampler(num_samples=num_samples)
    self.hash_encoder = HashGridEncoder(
        num_levels=num_levels, features_per_level=features_per_level)
    self.nerf_mlp = NeRFMLP(
        input_dim=num_levels * features_per_level, hidden_dim=hidden_dim)

  def forward(self, rays_o, rays_d):
    """Full NeRF forward pass.

    Args:
      rays_o: Ray origins [batch_size, 3].
      rays_d: Ray directions [batch_size, 3].

    Returns:
      Tuple of:
        density: Volume density [batch_size, num_samples, 1].
        rgb: RGB color [batch_size, num_samples, 3].
    """
    # 1. Sample positions along rays.
    positions = self.ray_sampler(rays_o, rays_d)  # [B, N, 3]

    # 2. Hash encoding.
    encoded = self.hash_encoder(positions)  # [B, N, L*F]

    # 3. MLP prediction.
    density, rgb = self.nerf_mlp(encoded, rays_d)

    return density, rgb


if __name__ == '__main__':
  print('=' * 70)
  print('NeRF Components Test')
  print('=' * 70)

  # Test RaySampler.
  print('\n1. Testing RaySampler...')
  sampler = RaySampler(num_samples=16)
  rays_o = torch.randn(2, 3)
  rays_d = torch.randn(2, 3)
  positions = sampler(rays_o, rays_d)
  print(f'✓ RaySampler output shape: {positions.shape}')

  # Test HashGridEncoder.
  print('\n2. Testing HashGridEncoder...')
  encoder = HashGridEncoder(
      num_levels=2, features_per_level=2, log2_hashmap_size=8)
  encoded = encoder(positions)
  print(f'✓ HashGridEncoder output shape: {encoded.shape}')

  # Test NeRFMLP.
  print('\n3. Testing NeRFMLP...')
  mlp = NeRFMLP(input_dim=4, hidden_dim=32)
  density, rgb = mlp(encoded, rays_d)
  print(f'✓ NeRFMLP density shape: {density.shape}')
  print(f'✓ NeRFMLP rgb shape: {rgb.shape}')

  # Test full model.
  print('\n4. Testing Complete Model...')
  model = HashGridNeRF(
      num_samples=16, num_levels=2, features_per_level=2, hidden_dim=32)
  density, rgb = model(rays_o, rays_d)
  print('✓ Complete model works!')
  print(f'  Density shape: {density.shape}')
  print(f'  RGB shape: {rgb.shape}')

  print('\n' + '=' * 70)
  print('All tests passed!')
  print('=' * 70)
