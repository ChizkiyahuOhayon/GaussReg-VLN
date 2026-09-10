import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'budgeted_factorized_landmark_for_test',
    ROOT / 'vlnce_baselines/budgeted_factorized_landmark.py',
)
BFLR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BFLR)


def tensors():
    base = torch.tensor([
        [0.0, -float('inf'), 2.0, 0.0],
        [0.0, -float('inf'), 0.0, 2.0],
    ])
    routed = torch.tensor([
        [0.0, -float('inf'), 0.0, 2.0],
        [0.0, -float('inf'), 2.0, 0.0],
    ])
    masks = torch.tensor([
        [False, False, True, True],
        [False, False, True, True],
    ])
    costs = torch.tensor([
        [0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.2, 0.8],
    ])
    return base, routed, masks, costs


def test_current_frontier_costs_use_latest_visited_node():
    step_ids = torch.tensor([[0, 1, 3, 0, 0]])
    visited = torch.tensor([[False, True, True, False, False]])
    pair_dists = torch.tensor([[
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.2, 0.3],
        [0.0, 0.1, 0.0, 0.4, 0.7],
        [0.0, 0.2, 0.4, 0.0, 0.5],
        [0.0, 0.3, 0.7, 0.5, 0.0],
    ]])

    actual = BFLR.current_frontier_costs(step_ids, pair_dists, visited)

    assert torch.equal(actual, pair_dists[:, 2])


def test_projection_is_exact_noop_when_routed_cost_is_within_budget():
    base, routed, masks, costs = tensors()
    projected, diagnostics = BFLR.project_frontier_logits(
        base[1:], routed[1:], masks[1:], costs[1:]
    )

    assert torch.equal(projected, routed[1:])
    assert diagnostics['activation_rate'].item() == 0.0
    assert diagnostics['dual_lambda'].item() == 0.0
    assert diagnostics['projected_cost'].item() == pytest.approx(
        diagnostics['routed_cost'].item()
    )


def test_projection_meets_e0_expected_cost_with_minimal_tilt():
    base, routed, masks, costs = tensors()
    projected, diagnostics = BFLR.project_frontier_logits(
        base[:1], routed[:1], masks[:1], costs[:1]
    )

    assert diagnostics['activation_rate'].item() == 1.0
    assert diagnostics['dual_lambda'].item() > 0.0
    assert diagnostics['routed_cost'] > diagnostics['base_cost']
    assert diagnostics['projected_cost'] <= diagnostics['base_cost'] + 1e-6
    assert torch.equal(projected[:, :2], routed[:1, :2])


def test_projection_handles_mixed_rows_and_equal_costs_without_nan():
    base, routed, masks, costs = tensors()
    costs[1, 2:] = 0.5
    projected, diagnostics = BFLR.project_frontier_logits(
        base, routed, masks, costs
    )

    assert torch.isfinite(projected[masks]).all()
    assert diagnostics['activation_rate'].item() == pytest.approx(0.5)
    assert diagnostics['dual_lambda'].item() > 0.0
    assert torch.equal(projected[1], routed[1])


def test_projection_preserves_frontier_gradients_when_budget_is_active():
    base, routed, masks, costs = tensors()
    routed = routed[:1].clone().requires_grad_(True)
    projected, _ = BFLR.project_frontier_logits(
        base[:1], routed, masks[:1], costs[:1]
    )
    log_probs = torch.log_softmax(
        projected.masked_fill(masks[:1].logical_not(), -float('inf')),
        dim=-1,
    )

    (-log_probs[0, 2]).backward()

    assert torch.isfinite(routed.grad).all()
    assert torch.count_nonzero(routed.grad[0, :2]) == 0
    assert torch.count_nonzero(routed.grad[0, 2:]) == 2


@pytest.mark.parametrize('failure', ['shape', 'mask', 'negative', 'nonfinite'])
def test_projection_rejects_invalid_route_evidence(failure):
    base, routed, masks, costs = tensors()
    if failure == 'shape':
        costs = costs[:, :-1]
    elif failure == 'mask':
        masks = masks.float()
    elif failure == 'negative':
        costs[0, 2] = -0.1
    else:
        costs[0, 2] = float('nan')

    with pytest.raises(ValueError):
        BFLR.project_frontier_logits(base, routed, masks, costs)
