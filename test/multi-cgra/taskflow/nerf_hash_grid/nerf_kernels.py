"""NeRF core compute kernels -- compiler-optimizable standalone modules.

Contains two parts, each matching the original CUDA logic one-to-one:

1. Sampling (ray marching + volume compositing)
   - march_rays:         Per-ray marching with occupancy grid skipping.
   - composite_rays:     Accumulates color/depth along samples.
   - near_far_from_aabb: Ray-AABB intersection.

2. Hash Encoding (multi-level hash grid encoding)
   - hash_encode:        Multi-level hash grid encoding + trilinear interp.

All functions are pure Python / NumPy / PyTorch with no CUDA dependency.
Designed for downstream compilation to parallel hardware.
"""

import math
import os
import sys
import numpy as np
import torch

# Adds the project-root python/ directory to sys.path and registers the
# neura::gather custom op.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'python'))
import neura_ops  # noqa: E402


# ================================================================
#  Part 1: Sampling -- Ray marching + volume compositing.
# ================================================================

# ---- Helper functions (scalar version, matches CUDA kernels) ----

def signf(x):
    """copysignf(1.0, x)"""
    return math.copysign(1.0, x)


def expand_bits(v):
    """Expands the lower 10 bits into Morton code format (2-bit spacing)."""
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v


def morton3D(x, y, z):
    """Computes 3D Morton (Z-order) code."""
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)


def mip_from_pos(x, y, z, max_cascade):
    """Determines the mip level from position."""
    mx = max(abs(x), abs(y), abs(z))
    if mx <= 0:
        exp = 0
    else:
        _, exp = math.frexp(mx)
    return int(min(max_cascade - 1, max(0, exp)))


def mip_from_dt(dt, H, max_cascade):
    """Determines the mip level from step size."""
    mx = dt * H * 0.5
    if mx <= 0:
        exp = 0
    else:
        _, exp = math.frexp(mx)
    return int(min(max_cascade - 1, max(0, exp)))


# ---- Ray-AABB intersection ----

def near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.2):
    """Computes ray-AABB intersection (matches CUDA kernel_near_far_from_aabb).

    Args:
        rays_o: Ray origins of shape [N, 3], float32.
        rays_d: Ray directions of shape [N, 3], float32.
        aabb: Axis-aligned bounding box [xmin,ymin,zmin,xmax,ymax,zmax].
        min_near: Minimum near-plane distance.

    Returns:
        Tuple of (nears, fars), each [N] float32.
    """
    FLT_MAX = torch.finfo(rays_o.dtype).max

    rdx = 1.0 / rays_d[:, 0]
    rdy = 1.0 / rays_d[:, 1]
    rdz = 1.0 / rays_d[:, 2]

    ox, oy, oz = rays_o[:, 0], rays_o[:, 1], rays_o[:, 2]

    # X
    near = (aabb[0] - ox) * rdx
    far  = (aabb[3] - ox) * rdx
    swap = near > far
    near, far = torch.where(swap, far, near), torch.where(swap, near, far)

    # Y
    near_y = (aabb[1] - oy) * rdy
    far_y  = (aabb[4] - oy) * rdy
    swap = near_y > far_y
    near_y, far_y = torch.where(swap, far_y, near_y), torch.where(swap, near_y, far_y)

    invalid = (near > far_y) | (near_y > far)
    near = torch.where(near_y > near, near_y, near)
    far  = torch.where(far_y  < far,  far_y,  far)

    # Z
    near_z = (aabb[2] - oz) * rdz
    far_z  = (aabb[5] - oz) * rdz
    swap = near_z > far_z
    near_z, far_z = torch.where(swap, far_z, near_z), torch.where(swap, near_z, far_z)

    invalid = invalid | (near > far_z) | (near_z > far)
    near = torch.where(near_z > near, near_z, near)
    far  = torch.where(far_z  < far,  far_z,  far)

    near = torch.where(invalid, torch.full_like(near, FLT_MAX), near)
    far  = torch.where(invalid, torch.full_like(far,  FLT_MAX), far)

    near = torch.clamp(near, min=min_near)
    return near, far


# ---- march_rays ----

def march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
               bound, density_bitfield, cascade, grid_size,
               nears, fars, dt_gamma, max_steps, perturb=False):
    """Performs per-ray marching (matches CUDA kernel_march_rays).

    Each alive ray advances forward, recording up to n_step samples in
    occupied regions. Uses density_bitfield to skip empty space.

    Args:
        n_alive: Number of alive rays.
        n_step: Maximum samples per ray.
        rays_alive: Alive ray indices, shape [>=n_alive], int32.
        rays_t: Current t per ray, shape [N_total], float32.
        rays_o: Ray origins, shape [N_total, 3], float32.
        rays_d: Ray directions, shape [N_total, 3], float32.
        bound: Scene bounding radius.
        density_bitfield: Occupancy bitfield, shape [M], uint8.
        cascade: Number of cascade levels.
        grid_size: Grid resolution (H, typically 128).
        nears: Near planes, shape [N_total], float32.
        fars: Far planes, shape [N_total], float32.
        dt_gamma: Adaptive step-size factor.
        max_steps: Maximum steps (used for dt_min).
        perturb: Whether to apply random perturbation.

    Returns:
        Tuple of (xyzs, dirs, deltas):
            xyzs:   Sample positions, shape [n_alive * n_step, 3], float32.
            dirs:   Sample directions, shape [n_alive * n_step, 3], float32.
            deltas: Step sizes, shape [n_alive * n_step, 2], float32.
                    deltas[:, 0] = dt (for density), deltas[:, 1] = real_delta (for depth).
    """
    SQRT3 = np.float32(1.7320508075688772)
    H = grid_size
    H3 = H * H * H

    _dt_min   = np.float32(2 * SQRT3 / max_steps)
    _dt_max   = np.float32(2 * SQRT3 * (1 << (cascade - 1)) / H)
    _rH       = np.float32(1.0 / H)
    _dt_gamma = np.float32(dt_gamma)
    _bound    = np.float32(bound)

    device = rays_o.device

    xyzs   = torch.zeros(n_alive * n_step, 3, device=device)
    dirs   = torch.zeros(n_alive * n_step, 3, device=device)
    deltas = torch.zeros(n_alive * n_step, 2, device=device)

    # Iterates over each ray (one ray = one CUDA thread).
    for n in range(n_alive):
        index = rays_alive[n].item()

        ox = np.float32(rays_o[index, 0].item())
        oy = np.float32(rays_o[index, 1].item())
        oz = np.float32(rays_o[index, 2].item())
        dx = np.float32(rays_d[index, 0].item())
        dy = np.float32(rays_d[index, 1].item())
        dz = np.float32(rays_d[index, 2].item())

        rdx = np.float32(1.0) / dx
        rdy = np.float32(1.0) / dy
        rdz = np.float32(1.0) / dz

        t       = np.float32(rays_t[index].item())
        far_val = np.float32(fars[index].item())

        if perturb:
            noise = np.float32(np.random.rand())
            t += np.float32(max(_dt_min, min(_dt_max, t * _dt_gamma))) * noise

        last_t = t
        step   = 0
        base   = n * n_step

        while t < far_val and step < n_step:
            x = np.float32(max(-_bound, min(_bound, ox + t * dx)))
            y = np.float32(max(-_bound, min(_bound, oy + t * dy)))
            z = np.float32(max(-_bound, min(_bound, oz + t * dz)))

            dt_val = np.float32(max(_dt_min, min(_dt_max, t * _dt_gamma)))

            level = max(
                mip_from_pos(float(x), float(y), float(z), cascade),
                mip_from_dt(float(dt_val), H, cascade),
            )

            mip_bound  = np.float32(min(2.0 ** level, float(_bound)))
            mip_rbound = np.float32(1.0) / mip_bound

            nx = int(max(0, min(H - 1, 0.5 * (x * mip_rbound + 1) * H)))
            ny = int(max(0, min(H - 1, 0.5 * (y * mip_rbound + 1) * H)))
            nz = int(max(0, min(H - 1, 0.5 * (z * mip_rbound + 1) * H)))

            bit_index = level * H3 + morton3D(nx, ny, nz)
            byte_idx  = bit_index // 8
            bit_pos   = bit_index % 8

            if byte_idx < density_bitfield.shape[0]:
                occ = (density_bitfield[byte_idx].item() >> bit_pos) & 1
            else:
                occ = 0

            if occ:
                idx = base + step
                xyzs[idx, 0]   = float(x)
                xyzs[idx, 1]   = float(y)
                xyzs[idx, 2]   = float(z)
                dirs[idx, 0]   = float(dx)
                dirs[idx, 1]   = float(dy)
                dirs[idx, 2]   = float(dz)
                t = np.float32(t + dt_val)
                deltas[idx, 0] = float(dt_val)
                deltas[idx, 1] = float(t - last_t)
                last_t = t
                step += 1
            else:
                tx = np.float32((((nx + 0.5 + 0.5 * signf(float(dx))) * _rH * 2 - 1) * mip_bound - x) * rdx)
                ty = np.float32((((ny + 0.5 + 0.5 * signf(float(dy))) * _rH * 2 - 1) * mip_bound - y) * rdy)
                tz = np.float32((((nz + 0.5 + 0.5 * signf(float(dz))) * _rH * 2 - 1) * mip_bound - z) * rdz)
                tt = np.float32(t + max(np.float32(0.0), min(tx, ty, tz)))
                # do-while (at least one step).
                t = np.float32(t + max(_dt_min, min(_dt_max, t * _dt_gamma)))
                while t < tt:
                    t = np.float32(t + max(_dt_min, min(_dt_max, t * _dt_gamma)))

    return xyzs, dirs, deltas


# ---- composite_rays ----

def composite_rays(n_alive, n_step, T_thresh, rays_alive, rays_t,
                   sigmas, rgbs, deltas,
                   weights_sum, depth, image):
    """Performs volume compositing (matches CUDA kernel_composite_rays).

    Accumulates color and depth along samples for each alive ray.
    Marks a ray as dead when T < T_thresh.

    Modifies in-place: rays_alive, rays_t, weights_sum, depth, image.

    Args:
        n_alive: Number of alive rays.
        n_step: Number of samples per ray.
        T_thresh: Transmittance threshold.
        rays_alive: Alive ray indices, shape [>=n_alive], int32.
        rays_t: Current t per ray, shape [N_total], float32.
        sigmas: Density values, shape [n_alive * n_step], float32.
        rgbs: Color values, shape [n_alive * n_step, 3], float32.
        deltas: Step sizes, shape [n_alive * n_step, 2], float32.
        weights_sum: Accumulated weights, shape [N_total], float32.
        depth: Accumulated depth, shape [N_total], float32.
        image: Accumulated color, shape [N_total, 3], float32.
    """
    for n in range(n_alive):
        index = rays_alive[n].item()
        base  = n * n_step

        t     = rays_t[index].item()
        w_sum = weights_sum[index].item()
        d     = depth[index].item()
        r     = image[index, 0].item()
        g     = image[index, 1].item()
        b     = image[index, 2].item()

        step = 0
        while step < n_step:
            idx = base + step

            if deltas[idx, 0].item() == 0:
                break

            alpha  = 1.0 - math.exp(-sigmas[idx].item() * deltas[idx, 0].item())
            T      = 1.0 - w_sum
            weight = alpha * T
            w_sum += weight

            t += deltas[idx, 1].item()
            d += weight * t
            r += weight * rgbs[idx, 0].item()
            g += weight * rgbs[idx, 1].item()
            b += weight * rgbs[idx, 2].item()

            if T < T_thresh:
                break

            step += 1

        if step < n_step:
            rays_alive[n] = -1
        else:
            rays_t[index] = t

        weights_sum[index] = w_sum
        depth[index]       = d
        image[index, 0]    = r
        image[index, 1]    = g
        image[index, 2]    = b


# ================================================================
#  Part 2: Hash Encoding -- Multi-level hash grid + trilinear interp.
# ================================================================

def hash_function(pos_grid, hashmap_size):
    """Computes 3D spatial hash using coherent hash (same primes as CUDA).

    Args:
        pos_grid: Integer grid coordinates of shape [..., 3], int64.
        hashmap_size: Hash-table size.

    Returns:
        Hash indices of shape [...], int64.
    """
    primes = torch.tensor([1, 2654435761, 805459861],
                           dtype=torch.int64, device=pos_grid.device)
    pos_grid = pos_grid.long()
    result = torch.zeros(pos_grid.shape[:-1], dtype=torch.int64, device=pos_grid.device)
    for i in range(3):
        result = torch.bitwise_xor(result, pos_grid[..., i] * primes[i])
    return (result % hashmap_size).long()


def get_grid_index(pos_grid, resolution, hashmap_size, input_dim=3,
                   gridtype='hash', align_corners=False):
    """Converts grid coordinates to embedding indices.

    Args:
        pos_grid: Grid coordinates of shape [..., 3], int64.
        resolution: Grid resolution.
        hashmap_size: Hash-table size.
        input_dim: Number of spatial dimensions.
        gridtype: ``'hash'`` or ``'tiled'``.
        align_corners: Whether to align corners.

    Returns:
        Embedding indices of shape [...], int64.
    """
    stride = 1
    index = torch.zeros(pos_grid.shape[:-1], dtype=torch.int64, device=pos_grid.device)
    for d in range(input_dim):
        index = index + pos_grid[..., d] * stride
        stride *= (resolution if align_corners else resolution + 1)

    if gridtype == 'hash' and stride > hashmap_size:
        index = hash_function(pos_grid, hashmap_size)
    else:
        index = index % hashmap_size

    return index.long()


def trilinear_interpolation(pos, pos_grid, level_embeddings,
                            resolution, hashmap_size,
                            gridtype='hash', align_corners=False):
    """Performs trilinear interpolation over a hash-grid embedding table.

    Two-phase implementation:
      Phase 1 -- Collects hash indices for all 8 corners into [8*N].
      Phase 2 -- Issues a single neura.gather to fetch all embeddings,
                 then accumulates trilinear-weighted features per corner.

    This produces exactly one neura.gather per level, matching the
    hardware gather semantics.

    Args:
        pos: Fractional positions of shape [N, 3], float32.
        pos_grid: Integer grid coordinates of shape [N, 3], int64.
        level_embeddings: Embedding table of shape [T, C], float32.
        resolution: Grid resolution for this level.
        hashmap_size: Hash-table size for this level.
        gridtype: Grid type, ``'hash'`` or ``'tiled'``.
        align_corners: Whether to align corners.

    Returns:
        Interpolated features of shape [N, C], float32.
    """
    device    = pos.device
    N         = pos.shape[0]
    level_dim = level_embeddings.shape[1]

    # Phase 1: Collects hash indices for all 8 corners -> [8*N].
    all_indices = []
    for idx in range(8):
        d0 = (idx >> 0) & 1
        d1 = (idx >> 1) & 1
        d2 = (idx >> 2) & 1

        corner = torch.stack([
            pos_grid[:, 0] + d0,
            pos_grid[:, 1] + d1,
            pos_grid[:, 2] + d2,
        ], dim=-1)

        indices = get_grid_index(corner, resolution, hashmap_size,
                                 gridtype=gridtype, align_corners=align_corners)
        indices = torch.clamp(indices, 0, level_embeddings.shape[0] - 1)
        all_indices.append(indices)                       # [N]

    batched_indices = torch.cat(all_indices, dim=0)       # [8*N]

    # Single gather fetches all corner embeddings at once.
    batched_features = torch.ops.neura.gather(
        level_embeddings, batched_indices)                # [8*N, C]

    # Phase 2: Computes trilinear weights and accumulates per corner.
    results = torch.zeros(N, level_dim, device=device, dtype=level_embeddings.dtype)

    for idx in range(8):
        d0 = (idx >> 0) & 1
        d1 = (idx >> 1) & 1
        d2 = (idx >> 2) & 1

        w0 = pos[:, 0] if d0 else (1 - pos[:, 0])
        w1 = pos[:, 1] if d1 else (1 - pos[:, 1])
        w2 = pos[:, 2] if d2 else (1 - pos[:, 2])
        weight = w0 * w1 * w2

        features = batched_features[idx * N : (idx + 1) * N]  # [N, C]
        results  = results + weight.unsqueeze(-1) * features

    return results


def hash_encode(inputs, embeddings, offsets, bound,
                num_levels=16, level_dim=2, base_resolution=16,
                per_level_scale=None, log2_hashmap_size=19,
                desired_resolution=None,
                gridtype='hash', align_corners=False):
    """Performs multi-level hash grid encoding (full forward pass).

    Matches CUDA GridEncoder::forward:
    1. Normalizes inputs to [0, 1].
    2. Per level: computes scale, maps grid coords, trilinear interpolation.
    3. Concatenates all level outputs.

    Args:
        inputs: Coordinates of shape [N, 3], float32, in [-bound, bound].
        embeddings: Hash-table params of shape [total_params, level_dim], float32.
        offsets: Per-level offsets of shape [num_levels + 1], int32.
        bound: Scene bounding radius.
        num_levels: Number of levels.
        level_dim: Output dimension per level.
        base_resolution: Base grid resolution.
        per_level_scale: Per-level scale factor (None derives from desired_resolution).
        log2_hashmap_size: Log2 of hash-table size.
        desired_resolution: Desired finest resolution.
        gridtype: ``'hash'`` or ``'tiled'``.
        align_corners: Whether to align corners.

    Returns:
        Encoded features of shape [N, num_levels * level_dim], float32.
    """
    if per_level_scale is None and desired_resolution is not None:
        per_level_scale = np.exp2(
            np.log2(desired_resolution / base_resolution) / (num_levels - 1)
        )

    # Normalizes to [0, 1].
    inputs = (inputs + bound) / (2 * bound)

    prefix_shape = list(inputs.shape[:-1])
    inputs = inputs.view(-1, 3)   # [N, 3]

    level_outputs = []

    for level in range(num_levels):
        # Scale computation matches CUDA:
        #   const float scale = exp2f(level * S) * H - 1.0f
        S = np.log2(per_level_scale)
        scale = np.exp2(level * S) * base_resolution - 1.0
        resolution = int(np.ceil(scale)) + 1

        offset_start = offsets[level].item()
        offset_end   = offsets[level + 1].item()
        hashmap_size = offset_end - offset_start
        level_emb    = embeddings[offset_start:offset_end]   # [T, C]

        # Grid coordinates (matches CUDA).
        #   pos[d] = inputs[d] * scale + (align_corners ? 0 : 0.5)
        pos_scaled = inputs * scale + (0.0 if align_corners else 0.5)
        pos_grid   = torch.floor(pos_scaled).long()
        pos        = pos_scaled - pos_grid.float()
        pos        = torch.clamp(pos, 0.0, 1.0)

        # Out-of-bounds check.
        oob = ((inputs < 0) | (inputs > 1)).any(dim=-1)

        features = trilinear_interpolation(
            pos, pos_grid, level_emb, resolution, hashmap_size,
            gridtype=gridtype, align_corners=align_corners,
        )
        features = torch.where(oob.unsqueeze(-1), torch.zeros_like(features), features)

        level_outputs.append(features)

    outputs = torch.cat(level_outputs, dim=-1)           # [N, num_levels * level_dim]
    outputs = outputs.view(prefix_shape + [num_levels * level_dim])
    return outputs