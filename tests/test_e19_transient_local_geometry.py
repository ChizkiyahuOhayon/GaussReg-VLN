import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] /
    'vlnce_baselines/transient_local_geometry.py'
)
SPEC = importlib.util.spec_from_file_location(
    'transient_local_geometry', MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TransientLocalGeometry = MODULE.TransientLocalGeometry
align_candidate_point_sets = MODULE.align_candidate_point_sets
depth_to_local_point_sets = MODULE.depth_to_local_point_sets


def test_depth_to_local_point_sets_is_deterministic_and_bounded():
    depth = torch.linspace(0.01, 0.5, 32 * 32).reshape(1, 32, 32)
    first, first_mask = depth_to_local_point_sets(
        depth, num_points=16, max_depth=3.0, stride=2
    )
    second, second_mask = depth_to_local_point_sets(
        depth, num_points=16, max_depth=3.0, stride=2
    )

    assert torch.equal(first_mask, torch.tensor([True]))
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first[..., 2].min() > 0
    assert first[..., 2].max() <= 1.0


def test_depth_to_local_point_sets_masks_empty_candidates():
    depth = torch.full((2, 16, 16), float('nan'))
    depth[1].fill_(0.9)
    points, masks = depth_to_local_point_sets(
        depth, num_points=8, max_depth=3.0, stride=2
    )

    assert torch.equal(masks, torch.tensor([False, False]))
    assert torch.count_nonzero(points) == 0


def test_candidate_features_align_and_average_duplicate_targets():
    features = [torch.tensor([[1.0, 3.0], [3.0, 5.0], [9.0, 9.0]])]
    masks = [torch.tensor([True, True, False])]
    aligned, aligned_masks = align_candidate_point_sets(
        [[None, '0', 'g0', 'g1']],
        [['g0', 'g0', 'g1']],
        features,
        masks,
    )

    assert torch.equal(aligned[0, 2], torch.tensor([2.0, 4.0]))
    assert torch.equal(aligned_masks, torch.tensor([[False, False, True, False]]))


def test_zero_initialized_geometry_is_exactly_inactive():
    module = TransientLocalGeometry(hidden_size=12, geometry_size=8)
    points = torch.randn(3, 16, 3)
    masks = torch.tensor([True, False, True])
    residual, diagnostics = module(points.unsqueeze(0), masks.unsqueeze(0))

    assert torch.count_nonzero(residual) == 0
    assert diagnostics['residual_norm'].item() == 0
    assert diagnostics['coverage'].item() == pytest.approx(2 / 3)


def test_geometry_has_finite_gradient_after_activation():
    module = TransientLocalGeometry(hidden_size=12, geometry_size=8)
    module.output.weight.data.normal_(std=0.01)
    points = torch.randn(3, 16, 3)
    masks = torch.tensor([True, False, True])
    residual, _ = module(points.unsqueeze(0), masks.unsqueeze(0))
    residual.square().mean().backward()

    assert torch.count_nonzero(residual[:, 1]) == 0
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
