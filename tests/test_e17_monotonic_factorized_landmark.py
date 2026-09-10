import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ModuleType('vlnce_baselines')
PACKAGE.__path__ = [str(ROOT / 'vlnce_baselines')]
sys.modules.setdefault('vlnce_baselines', PACKAGE)
LTG_SPEC = importlib.util.spec_from_file_location(
    'vlnce_baselines.landmark_transport',
    ROOT / 'vlnce_baselines/landmark_transport.py',
)
LTG = importlib.util.module_from_spec(LTG_SPEC)
sys.modules.setdefault('vlnce_baselines.landmark_transport', LTG)
LTG_SPEC.loader.exec_module(LTG)
SPEC = importlib.util.spec_from_file_location(
    'monotonic_factorized_landmark_for_test',
    ROOT / 'vlnce_baselines/monotonic_factorized_landmark.py',
)
MFLR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MFLR)


def test_first_observation_starts_at_first_valid_slot():
    emissions = torch.tensor([[[0.0, 100.0, 100.0, 100.0]]])
    history = torch.tensor([[True]])
    slots = torch.tensor([[True, True, True, True]])

    progress = MFLR.monotonic_progress(emissions, history, slots)

    assert torch.equal(progress, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_alignment_can_only_stay_or_advance_one_slot_per_observation():
    emissions = torch.tensor([[
        [5.0, -5.0, -5.0, -5.0],
        [-5.0, 5.0, -5.0, -5.0],
        [-5.0, -5.0, 5.0, -5.0],
    ]])
    progress = MFLR.monotonic_progress(
        emissions,
        torch.tensor([[True, True, True]]),
        torch.tensor([[True, True, True, True]]),
    )

    assert progress.argmax(dim=-1).item() == 2
    assert progress[0, 3].item() == 0.0
    assert progress.sum().item() == pytest.approx(1.0)


def test_alignment_supports_noncontiguous_valid_slots_and_padding():
    emissions = torch.tensor([[
        [4.0, 99.0, -4.0, 99.0],
        [-4.0, 99.0, 4.0, 99.0],
        [99.0, 99.0, 99.0, 99.0],
    ]])
    progress = MFLR.monotonic_progress(
        emissions,
        torch.tensor([[True, True, False]]),
        torch.tensor([[True, False, True, False]]),
    )

    assert progress.argmax(dim=-1).item() == 2
    assert torch.count_nonzero(progress[0, [1, 3]]).item() == 0


def test_empty_history_falls_back_to_first_valid_slot():
    progress = MFLR.monotonic_progress(
        torch.zeros(1, 2, 4),
        torch.tensor([[False, False]]),
        torch.tensor([[False, True, True, False]]),
    )
    assert torch.equal(progress, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))


def test_alignment_is_differentiable_and_rejects_nonfinite_evidence():
    emissions = torch.tensor([[
        [1.0, -1.0], [-1.0, 1.0],
    ]], requires_grad=True)
    history = torch.tensor([[True, True]])
    slots = torch.tensor([[True, True]])
    progress = MFLR.monotonic_progress(emissions, history, slots)
    progress[0, 1].backward()

    assert torch.isfinite(emissions.grad).all()
    assert emissions.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='must be finite'):
        MFLR.monotonic_progress(
            emissions.detach().fill_(float('nan')), history, slots
        )


def test_graph_history_is_sorted_and_frontiers_are_excluded():
    emissions = torch.tensor([[
        [0.0], [20.0], [10.0], [999.0],
    ]])
    masks = torch.tensor([[False, True, True, False]])
    steps = torch.tensor([[0, 2, 1, 0]])

    ordered, ordered_masks = MFLR.ordered_history(
        emissions, masks, steps
    )

    assert ordered[0, :2, 0].tolist() == [10.0, 20.0]
    assert ordered_masks.tolist() == [[True, True, False, False]]


def test_monotonic_transport_keeps_e15_parameter_budget_and_zero_output():
    torch.manual_seed(17)
    module = MFLR.MonotonicLandmarkTransport(16, 8)
    views = torch.randn(2, 5, 3, 16)
    view_masks = torch.tensor([
        [[False] * 3, [True] * 3, [True, False, False],
         [True, True, False], [True, False, False]],
        [[False] * 3, [True, True, False], [True, False, False],
         [True, False, False], [False] * 3],
    ])
    residual, diagnostics = module(
        views,
        view_masks,
        torch.randn(2, 8, 16),
        torch.tensor([
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False, False],
        ]),
        torch.tensor([[0, 1, 2, 0, 0], [0, 1, 0, 0, 0]]),
        torch.tensor([
            [False, True, True, False, False],
            [False, True, False, False, False],
        ]),
    )

    assert torch.count_nonzero(residual).item() == 0
    assert diagnostics['stage_progress'].shape == (2, 4)
    assert torch.allclose(
        diagnostics['stage_progress'].sum(dim=-1), torch.ones(2)
    )
    assert sum(p.numel() for p in module.parameters()) == 800
