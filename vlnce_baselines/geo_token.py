import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_SECTORS = 3
FEATURE_SIZE = 8


def _camera_grid(height, width, hfov_degrees, device, dtype):
    hfov = math.radians(float(hfov_degrees))
    focal = (width * 0.5) / math.tan(hfov * 0.5)
    x = (
        torch.arange(width, device=device, dtype=dtype) -
        (width - 1.0) * 0.5
    ) / focal
    y = (
        torch.arange(height, device=device, dtype=dtype) -
        (height - 1.0) * 0.5
    ) / focal
    return (
        y.unsqueeze(1).expand(height, width),
        x.unsqueeze(0).expand(height, width),
    )


def gaussian_free_space_tokens(
    depth,
    heading_offsets,
    waypoint_distances,
    max_depth=10.0,
    hfov_degrees=90.0,
    stride=8,
):
    """Compress candidate-aligned depth into three Gaussian moment tokens.

    Each token describes one horizontal sector with support, free-space
    clearance, 3D mean, and diagonal variance. Depth is expected to be
    normalized to ``[0, 1]`` as configured by the VLN-CE depth sensor.
    """
    if depth.ndim != 3:
        raise ValueError('depth must have shape (candidates, height, width)')
    if heading_offsets.shape != depth.shape[:1]:
        raise ValueError('heading_offsets must match the candidate count')
    if waypoint_distances.shape != depth.shape[:1]:
        raise ValueError('waypoint_distances must match the candidate count')
    if stride < 1:
        raise ValueError('stride must be positive')

    sampled = depth[:, ::stride, ::stride].to(dtype=torch.float32)
    candidate_count, height, width = sampled.shape
    grid_y, grid_x = _camera_grid(
        height, width, hfov_degrees, sampled.device, sampled.dtype
    )
    finite_depth = torch.isfinite(sampled)
    safe_depth = torch.where(finite_depth, sampled, torch.zeros_like(sampled))
    z = safe_depth * float(max_depth)
    x = grid_x.unsqueeze(0) * z
    y = grid_y.unsqueeze(0) * z

    cosine = torch.cos(heading_offsets.float()).view(-1, 1, 1)
    sine = torch.sin(heading_offsets.float()).view(-1, 1, 1)
    relative_x = cosine * x + sine * z
    relative_z = -sine * x + cosine * z
    relative_angle = torch.atan2(relative_x, relative_z)

    finite = (
        finite_depth & torch.isfinite(relative_x) &
        torch.isfinite(y) & torch.isfinite(relative_z)
    )
    valid = finite & (sampled > 0.0) & (sampled <= 1.0) & (relative_z > 0.0)
    sector_edges = torch.tensor(
        [-math.pi / 4, -math.pi / 12, math.pi / 12, math.pi / 4],
        device=sampled.device,
        dtype=sampled.dtype,
    )

    tokens = sampled.new_zeros(candidate_count, NUM_SECTORS, FEATURE_SIZE)
    token_masks = torch.zeros(
        candidate_count, NUM_SECTORS, device=sampled.device, dtype=torch.bool
    )
    points = torch.stack([relative_x, y, relative_z], dim=-1)
    distance_scale = float(max_depth)
    variance_scale = distance_scale * distance_scale

    for sector in range(NUM_SECTORS):
        ray_mask = (
            (relative_angle >= sector_edges[sector]) &
            (relative_angle < sector_edges[sector + 1])
        )
        sample_mask = ray_mask & valid
        ray_count = ray_mask.flatten(1).sum(dim=1).clamp_min(1)
        sample_count = sample_mask.flatten(1).sum(dim=1)
        token_masks[:, sector] = sample_count > 0

        weights = sample_mask.unsqueeze(-1).to(dtype=points.dtype)
        count = sample_count.clamp_min(1).to(points.dtype).view(-1, 1)
        mean = (points * weights).sum(dim=(1, 2)) / count
        centered = points - mean[:, None, None, :]
        variance = (centered.square() * weights).sum(dim=(1, 2)) / count
        clearance = (
            sample_mask &
            (relative_z >= waypoint_distances.float().view(-1, 1, 1))
        ).flatten(1).sum(dim=1).to(points.dtype) / count.squeeze(1)

        features = torch.cat([
            (sample_count.float() / ray_count.float()).unsqueeze(1),
            clearance.unsqueeze(1),
            mean / distance_scale,
            variance / variance_scale,
        ], dim=1)
        tokens[:, sector] = features * token_masks[:, sector, None]

    return tokens, token_masks


def align_candidate_tokens(
    graph_viewpoint_ids,
    candidate_viewpoint_ids,
    candidate_tokens,
    candidate_masks,
):
    """Align current candidate tokens to padded graph-map action slots."""
    if not (
        len(graph_viewpoint_ids) == len(candidate_viewpoint_ids) ==
        len(candidate_tokens) == len(candidate_masks)
    ):
        raise ValueError('batched candidate-token inputs are misaligned')

    batch_size = len(graph_viewpoint_ids)
    max_graph_size = max(len(ids) for ids in graph_viewpoint_ids)
    reference = candidate_tokens[0]
    aligned = reference.new_zeros(
        batch_size, max_graph_size, NUM_SECTORS, FEATURE_SIZE
    )
    aligned_masks = torch.zeros(
        batch_size, max_graph_size, NUM_SECTORS,
        device=reference.device, dtype=torch.bool,
    )

    for batch_index, graph_ids in enumerate(graph_viewpoint_ids):
        graph_index = {
            viewpoint_id: index for index, viewpoint_id in enumerate(graph_ids)
            if viewpoint_id is not None and viewpoint_id.startswith('g')
        }
        sums = {}
        counts = {}
        for candidate_index, viewpoint_id in enumerate(
                candidate_viewpoint_ids[batch_index]):
            if viewpoint_id not in graph_index:
                continue
            token_mask = candidate_masks[batch_index][candidate_index]
            weighted_token = candidate_tokens[batch_index][candidate_index]
            weighted_token = weighted_token * token_mask[:, None]
            sums[viewpoint_id] = sums.get(viewpoint_id, 0) + weighted_token
            counts[viewpoint_id] = counts.get(
                viewpoint_id,
                torch.zeros_like(token_mask, dtype=weighted_token.dtype),
            ) + token_mask.to(dtype=weighted_token.dtype)
        for viewpoint_id, token_sum in sums.items():
            index = graph_index[viewpoint_id]
            token_count = counts[viewpoint_id]
            aligned[batch_index, index] = (
                token_sum / token_count.clamp_min(1)[:, None]
            )
            aligned_masks[batch_index, index] = token_count > 0

    return aligned, aligned_masks


class GeoTokenResidual(nn.Module):
    """Predict a goal-free frontier residual from transient geometry tokens."""

    def __init__(self, hidden_size):
        super().__init__()
        input_size = NUM_SECTORS * (FEATURE_SIZE + 1)
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens, token_masks, valid_mask, visited_mask):
        if tokens.shape[-2:] != (NUM_SECTORS, FEATURE_SIZE):
            raise ValueError('unexpected GeoToken feature shape')
        if token_masks.shape != tokens.shape[:-1]:
            raise ValueError('GeoToken values and masks are misaligned')
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError('GeoToken and graph masks are misaligned')
        if visited_mask.shape != valid_mask.shape:
            raise ValueError('graph valid and visited masks are misaligned')

        masks = token_masks.to(dtype=tokens.dtype)
        inputs = torch.cat([
            tokens.detach(), masks.unsqueeze(-1)
        ], dim=-1).flatten(start_dim=2)
        residual = self.output(F.gelu(self.hidden(inputs))).squeeze(-1)
        output_mask = valid_mask & visited_mask.logical_not()
        output_mask &= token_masks.any(dim=-1)
        output_mask[:, 0] = False
        return residual * output_mask.to(dtype=residual.dtype)
