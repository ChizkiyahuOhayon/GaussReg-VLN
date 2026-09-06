import importlib.util
import math
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / 'vlnce_baselines/geo_token.py'
)
SPEC = importlib.util.spec_from_file_location('geo_token_for_test', MODULE_PATH)
GEO_TOKEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEO_TOKEN)


def test_constant_depth_produces_finite_symmetric_moments():
    depth = torch.full((1, 9, 9), 0.5)
    tokens, masks = GEO_TOKEN.gaussian_free_space_tokens(
        depth,
        heading_offsets=torch.tensor([0.0]),
        waypoint_distances=torch.tensor([2.0]),
        stride=1,
    )

    assert masks.all()
    assert torch.isfinite(tokens).all()
    assert torch.allclose(tokens[0, 0, 1], torch.tensor(1.0))
    assert torch.allclose(tokens[0, 2, 1], torch.tensor(1.0))
    assert torch.allclose(tokens[0, 0, 2], -tokens[0, 2, 2], atol=1e-6)
    assert torch.allclose(tokens[0, 0, 4], tokens[0, 2, 4], atol=1e-6)


def test_heading_offset_rotates_candidate_coordinates():
    depth = torch.zeros(2, 9, 9)
    depth[:, 4, 4] = 0.5
    tokens, masks = GEO_TOKEN.gaussian_free_space_tokens(
        depth,
        heading_offsets=torch.tensor([0.0, math.pi / 12]),
        waypoint_distances=torch.tensor([2.0, 2.0]),
        stride=1,
    )

    assert not torch.allclose(tokens[0], tokens[1])
    assert torch.allclose(tokens[0, 1, 2], torch.tensor(0.0), atol=1e-6)
    assert tokens[1, :, 2].max() > 0.0
    assert masks.sum() == 2


def test_invalid_depth_is_masked_without_nan():
    depth = torch.zeros(1, 8, 8)
    depth[:, 0, 0] = float('nan')
    tokens, masks = GEO_TOKEN.gaussian_free_space_tokens(
        depth,
        heading_offsets=torch.tensor([0.0]),
        waypoint_distances=torch.tensor([1.0]),
        stride=1,
    )

    assert not masks.any()
    assert torch.count_nonzero(tokens) == 0
    assert torch.isfinite(tokens).all()


def test_candidate_alignment_averages_merged_ghosts():
    candidate_tokens = [torch.stack([
        torch.ones(3, 8),
        torch.full((3, 8), 3.0),
        torch.full((3, 8), 7.0),
    ])]
    candidate_masks = [torch.ones(3, 3, dtype=torch.bool)]
    aligned, masks = GEO_TOKEN.align_candidate_tokens(
        [[None, '0', 'g0', 'g1']],
        [['g0', 'g0', '0']],
        candidate_tokens,
        candidate_masks,
    )

    assert torch.equal(aligned[0, 2], torch.full((3, 8), 2.0))
    assert torch.count_nonzero(aligned[0, :2]) == 0
    assert masks[0, 2].all()
    assert not masks[0, :2].any()


def test_candidate_alignment_ignores_missing_sector_observations():
    candidate_tokens = [torch.stack([
        torch.full((3, 8), 2.0),
        torch.full((3, 8), 6.0),
    ])]
    candidate_masks = [torch.tensor([
        [True, False, True],
        [True, True, False],
    ])]
    aligned, masks = GEO_TOKEN.align_candidate_tokens(
        [[None, 'g0']],
        [['g0', 'g0']],
        candidate_tokens,
        candidate_masks,
    )

    assert torch.equal(aligned[0, 1, 0], torch.full((8,), 4.0))
    assert torch.equal(aligned[0, 1, 1], torch.full((8,), 6.0))
    assert torch.equal(aligned[0, 1, 2], torch.full((8,), 2.0))
    assert masks[0, 1].all()


def test_zero_residual_preserves_logits_and_detaches_geometry():
    module = GEO_TOKEN.GeoTokenResidual(hidden_size=16)
    tokens = torch.randn(2, 4, 3, 8, requires_grad=True)
    token_masks = torch.ones(2, 4, 3, dtype=torch.bool)
    valid = torch.ones(2, 4, dtype=torch.bool)
    visited = torch.tensor([
        [False, True, False, False],
        [False, False, False, False],
    ])

    residual = module(tokens, token_masks, valid, visited)
    assert torch.count_nonzero(residual) == 0
    assert torch.count_nonzero(residual[:, 0]) == 0
    assert torch.count_nonzero(residual[0, 1]) == 0

    residual.sum().backward()
    assert tokens.grad is None
    assert torch.count_nonzero(module.output.weight.grad) > 0
