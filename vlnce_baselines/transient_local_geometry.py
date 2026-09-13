"""Transient local geometry for currently observable frontier candidates."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _farthest_point_sample(points, count):
    if points.size(0) <= count:
        indices = torch.arange(count, device=points.device) % points.size(0)
        return points[indices]

    selected = torch.empty(count, dtype=torch.long, device=points.device)
    center = points.mean(dim=0, keepdim=True)
    selected[0] = (points - center).square().sum(dim=-1).argmax()
    distances = torch.full(
        (points.size(0),), float('inf'), device=points.device
    )
    for index in range(1, count):
        latest = points[selected[index - 1]]
        distances = torch.minimum(
            distances, (points - latest).square().sum(dim=-1)
        )
        selected[index] = distances.argmax()
    return points[selected]


def depth_to_local_point_sets(
    depth,
    num_points=128,
    max_depth=3.0,
    sensor_max_depth=10.0,
    hfov_degrees=90.0,
    stride=8,
):
    """Convert normalized candidate depth maps into bounded local point sets."""
    if depth.ndim != 3:
        raise ValueError('depth must have shape (candidates, height, width)')
    if num_points < 1 or stride < 1:
        raise ValueError('num_points and stride must be positive')
    if max_depth <= 0 or sensor_max_depth <= 0:
        raise ValueError('depth limits must be positive')

    sampled = depth[:, ::stride, ::stride].detach().float()
    candidate_count, height, width = sampled.shape
    hfov = math.radians(float(hfov_degrees))
    focal = (width * 0.5) / math.tan(hfov * 0.5)
    horizontal = (
        torch.arange(width, device=sampled.device, dtype=sampled.dtype) -
        (width - 1.0) * 0.5
    ) / focal
    vertical = (
        torch.arange(height, device=sampled.device, dtype=sampled.dtype) -
        (height - 1.0) * 0.5
    ) / focal

    z = sampled * float(sensor_max_depth)
    x = z * horizontal.view(1, 1, width)
    y = z * vertical.view(1, height, 1)
    points = torch.stack([x, y, z], dim=-1)
    valid = torch.isfinite(points).all(dim=-1)
    valid &= (z > 0.0) & (z <= float(max_depth))

    output = sampled.new_zeros(candidate_count, num_points, 3)
    masks = torch.zeros(
        candidate_count, device=sampled.device, dtype=torch.bool
    )
    for candidate_index in range(candidate_count):
        candidate_points = points[candidate_index][valid[candidate_index]]
        if candidate_points.numel() == 0:
            continue
        output[candidate_index] = _farthest_point_sample(
            candidate_points, num_points
        ) / float(max_depth)
        masks[candidate_index] = True
    return output, masks


def align_candidate_point_sets(
    graph_viewpoint_ids,
    candidate_viewpoint_ids,
    candidate_point_sets,
    candidate_masks,
):
    """Average duplicate candidate point sets into padded graph slots."""
    if not (
        len(graph_viewpoint_ids) == len(candidate_viewpoint_ids) ==
        len(candidate_point_sets) == len(candidate_masks)
    ):
        raise ValueError('batched candidate geometry inputs are misaligned')
    if not graph_viewpoint_ids:
        raise ValueError('candidate geometry batch must be non-empty')

    batch_size = len(graph_viewpoint_ids)
    max_graph_size = max(len(ids) for ids in graph_viewpoint_ids)
    reference = candidate_point_sets[0]
    aligned = reference.new_zeros(
        (batch_size, max_graph_size) + tuple(reference.shape[1:])
    )
    aligned_masks = torch.zeros(
        batch_size, max_graph_size,
        device=reference.device, dtype=torch.bool,
    )

    for batch_index, graph_ids in enumerate(graph_viewpoint_ids):
        graph_index = {
            viewpoint_id: index for index, viewpoint_id in enumerate(graph_ids)
            if viewpoint_id is not None
        }
        sums = {}
        counts = {}
        for candidate_index, viewpoint_id in enumerate(
                candidate_viewpoint_ids[batch_index]):
            if (viewpoint_id not in graph_index or
                    not candidate_masks[batch_index][candidate_index]):
                continue
            points = candidate_point_sets[batch_index][candidate_index]
            sums[viewpoint_id] = sums.get(viewpoint_id, 0) + points
            counts[viewpoint_id] = counts.get(viewpoint_id, 0) + 1
        for viewpoint_id, point_sum in sums.items():
            index = graph_index[viewpoint_id]
            aligned[batch_index, index] = (
                point_sum / counts[viewpoint_id]
            )
            aligned_masks[batch_index, index] = True
    return aligned, aligned_masks


class TransientLocalGeometry(nn.Module):
    """Encode local point sets and emit a zero-initialized graph residual."""

    def __init__(self, hidden_size, geometry_size):
        super().__init__()
        if geometry_size <= 0:
            raise ValueError('geometry_size must be positive')
        self.geometry_size = geometry_size
        self.point_projection = nn.Linear(3, geometry_size)
        self.point_refinement = nn.Linear(geometry_size, geometry_size)
        self.output = nn.Linear(geometry_size, hidden_size)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode(self, point_sets, candidate_masks):
        if point_sets.ndim != 3 or point_sets.size(-1) != 3:
            raise ValueError('point_sets must have shape (candidates, points, 3)')
        if candidate_masks.shape != point_sets.shape[:1]:
            raise ValueError('point sets and candidate masks are misaligned')
        if not torch.isfinite(point_sets).all():
            raise ValueError('local point sets must be finite')

        features = F.gelu(self.point_projection(point_sets))
        features = F.gelu(self.point_refinement(features))
        features = features.max(dim=1).values
        return features * candidate_masks.unsqueeze(-1).to(features.dtype)

    def forward(self, point_sets, feature_masks):
        if point_sets.ndim != 4 or point_sets.size(-1) != 3:
            raise ValueError('point_sets must have shape (batch, graph, points, 3)')
        if feature_masks.shape != point_sets.shape[:2]:
            raise ValueError('local point sets and masks are misaligned')
        batch_size, graph_size = point_sets.shape[:2]
        features = self.encode(
            point_sets.flatten(0, 1), feature_masks.flatten()
        ).view(batch_size, graph_size, self.geometry_size)
        residual = self.output(features)
        residual = residual * feature_masks.unsqueeze(-1).to(residual.dtype)
        active = feature_masks.any()
        diagnostics = {
            'coverage': feature_masks.float().mean().detach(),
            'residual_norm': (
                residual[feature_masks].norm(dim=-1).mean().detach()
                if active else residual.new_zeros(())
            ),
        }
        return residual, diagnostics
