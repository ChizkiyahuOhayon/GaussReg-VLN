import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
LTG_SPEC = importlib.util.spec_from_file_location(
    'landmark_transport_for_e15_test',
    ROOT / 'vlnce_baselines/landmark_transport.py',
)
LTG = importlib.util.module_from_spec(LTG_SPEC)
LTG_SPEC.loader.exec_module(LTG)
SPEC = importlib.util.spec_from_file_location(
    'factorized_landmark_for_test',
    ROOT / 'vlnce_baselines/factorized_landmark.py',
)
FLR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FLR)


def test_frontier_mask_excludes_stop_visited_and_padding():
    action_masks = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, False, False],
    ])
    visited_masks = torch.tensor([
        [False, True, False, False, False],
        [False, True, False, False, False],
    ])

    assert FLR.frontier_action_mask(action_masks, visited_masks).tolist() == [
        [False, False, True, True, False],
        [False, False, False, False, False],
    ]


def test_zero_transport_exactly_recovers_e0_distribution():
    base = torch.tensor([
        [1.2, -float('inf'), 0.8, -0.3, -float('inf')],
        [0.1, -float('inf'), 1.4, 0.7, -float('inf')],
    ])
    mask = torch.tensor([
        [False, False, True, True, False],
        [False, False, True, True, False],
    ])
    actual = FLR.factorized_action_log_probs(base, base, mask)
    expected = torch.log_softmax(base, dim=-1)

    finite = torch.isfinite(expected)
    assert torch.allclose(actual[finite], expected[finite], atol=1e-7)
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    assert torch.allclose(actual.exp().sum(dim=-1), torch.ones(2))


def test_factorization_preserves_stop_mass_and_redistributes_continue_mass():
    base = torch.tensor([[0.2, -float('inf'), 1.0, 0.8]])
    routed = torch.tensor([[-float('inf'), -float('inf'), -2.0, 3.0]])
    mask = torch.tensor([[False, False, True, True]])

    log_probs = FLR.factorized_action_log_probs(base, routed, mask)
    base_probs = torch.softmax(base, dim=-1)
    probs = log_probs.exp()

    assert probs[0, 0] == pytest.approx(base_probs[0, 0].item())
    assert probs[0, 2:].sum() == pytest.approx(base_probs[0, 2:].sum().item())
    assert probs[0, 3] > probs[0, 2]
    assert torch.isneginf(log_probs[0, 1])


def test_greedy_keeps_e0_stop_decision_and_can_reroute_continue():
    base = torch.tensor([
        [3.0, -float('inf'), 2.0, 1.0],
        [0.0, -float('inf'), 3.0, 2.0],
    ])
    routed = torch.tensor([
        [-float('inf'), -float('inf'), -5.0, 9.0],
        [-float('inf'), -float('inf'), -5.0, 9.0],
    ])
    mask = torch.tensor([
        [False, False, True, True],
        [False, False, True, True],
    ])

    actions = FLR.factorized_greedy_actions(base, routed, mask)
    assert actions.tolist() == [0, 3]


def test_no_frontier_falls_back_to_e0_stop():
    base = torch.tensor([[2.0, -float('inf'), -float('inf')]])
    mask = torch.zeros_like(base, dtype=torch.bool)
    actual = FLR.factorized_action_log_probs(base, base, mask)

    assert torch.equal(actual, torch.log_softmax(base, dim=-1))
    assert FLR.factorized_greedy_actions(base, base, mask).item() == 0
    assert FLR.conditional_frontier_kl(base, base, mask).item() == 0


def test_conditional_objective_has_finite_frontier_only_gradients():
    base = torch.tensor([[0.2, -float('inf'), 1.0, 0.8]])
    routed = torch.tensor(
        [[0.0, 0.0, -0.4, 0.6]], requires_grad=True
    )
    mask = torch.tensor([[False, False, True, True]])

    log_probs = FLR.factorized_action_log_probs(base, routed, mask)
    loss = -log_probs[0, 3] + FLR.conditional_frontier_kl(
        base, routed, mask
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(routed.grad).all()
    assert torch.count_nonzero(routed.grad[0, :2]) == 0
    assert torch.count_nonzero(routed.grad[0, 2:]) == 2


def test_factorized_module_keeps_e14_parameter_budget():
    module = LTG.LandmarkTransport(768, 128)
    assert sum(parameter.numel() for parameter in module.parameters()) == 590848
