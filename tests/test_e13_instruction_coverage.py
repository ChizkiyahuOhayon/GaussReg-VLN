import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'instruction_coverage_for_test',
    ROOT / 'vlnce_baselines/instruction_coverage.py',
)
IIEG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IIEG)


@pytest.fixture(scope='module')
def graph_utils():
    habitat = types.ModuleType('habitat')
    tasks = types.ModuleType('habitat.tasks')
    task_utils = types.ModuleType('habitat.tasks.utils')
    task_utils.cartesian_to_polar = lambda x, y: (
        np.hypot(x, y), np.arctan2(y, x)
    )
    utils = types.ModuleType('habitat.utils')
    geometry = types.ModuleType('habitat.utils.geometry_utils')
    geometry.quaternion_rotate_vector = lambda quaternion, vector: vector
    geometry.quaternion_from_coeff = lambda coefficients: coefficients
    modules = {
        'habitat': habitat,
        'habitat.tasks': tasks,
        'habitat.tasks.utils': task_utils,
        'habitat.utils': utils,
        'habitat.utils.geometry_utils': geometry,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            'graph_utils_for_e13',
            ROOT / 'vlnce_baselines/models/graph_utils.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.heading_from_quaternion = lambda quaternion: 0.0
        yield module
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def test_ordered_slots_partition_valid_tokens_and_ignore_padding():
    masks = torch.tensor([
        [True, True, True, True, True, True, True, True],
        [True, True, False, False, False, False, False, False],
    ])
    assignments, slot_masks = IIEG.ordered_slot_assignments(masks)

    assert assignments.sum(dim=-1).eq(masks).all()
    assert assignments[0].long().argmax(dim=-1).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert assignments[1, :2].long().argmax(dim=-1).tolist() == [0, 2]
    assert slot_masks.tolist() == [[True] * 4, [True, False, True, False]]
    assert not assignments[1, 2:].any()


def test_empty_instruction_is_rejected():
    with pytest.raises(ValueError, match='at least one token'):
        IIEG.ordered_slot_assignments(torch.zeros(1, 3, dtype=torch.bool))


def test_slot_evidence_is_normalized_and_padding_invariant():
    scores = torch.tensor([[[[3.0, 1.0, -2.0, 1000.0],
                             [0.0, 0.0, 0.0, -1000.0]]]])
    masks = torch.tensor([[True, True, True, False]])
    evidence, slot_masks = IIEG.instruction_slot_evidence(scores, masks)
    changed = scores.clone()
    changed[..., -1] = -5000.0
    other, _ = IIEG.instruction_slot_evidence(changed, masks)

    assert evidence.shape == (1, 2, 4)
    assert torch.allclose(evidence.sum(dim=-1), torch.ones(1, 2))
    assert torch.allclose(evidence, other)
    assert slot_masks.tolist() == [[True, True, True, False]]


def test_view_aggregation_is_idempotent_and_ignores_padding():
    evidence = torch.tensor([[[0.1, 0.4, 0.2, 0.3],
                              [0.5, 0.2, 0.3, 0.1],
                              [9.0, 9.0, 9.0, 9.0]]])
    masks = torch.tensor([[True, True, False]])
    pooled = IIEG.aggregate_view_evidence(evidence, masks)
    duplicated = IIEG.aggregate_view_evidence(
        torch.cat([evidence[:, :2], evidence[:, :1]], dim=1),
        torch.ones(1, 3, dtype=torch.bool),
    )

    assert torch.equal(pooled, torch.tensor([[0.5, 0.4, 0.3, 0.3]]))
    assert torch.equal(pooled, duplicated)


def test_coverage_is_monotonic_and_repeated_evidence_has_no_new_state():
    evidence = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                              [0.7, 0.2, 0.1, 0.0],
                              [0.7, 0.8, 0.1, 0.0],
                              [0.9, 0.4, 0.2, 0.5]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    first = torch.tensor([[False, True, False, False]])
    second = torch.tensor([[False, True, True, False]])

    coverage_a, marginal_a = IIEG.coverage_features(evidence, valid, first)
    coverage_b, marginal_b = IIEG.coverage_features(evidence, valid, second)

    assert torch.all(coverage_b >= coverage_a)
    assert torch.allclose(coverage_a, torch.tensor([[0.7, 0.2, 0.1, 0.0]]))
    assert torch.allclose(coverage_b, torch.tensor([[0.7, 0.8, 0.1, 0.0]]))
    assert marginal_a[0, 2, 0] == marginal_a[0, 1, 0]
    assert marginal_b[0, 2, 1] < marginal_a[0, 2, 1]


def test_zero_initialized_residual_preserves_e0_and_detaches_evidence():
    module = IIEG.InstructionCoverageResidual(hidden_size=32)
    evidence = torch.rand(2, 5, 4, requires_grad=True)
    valid = torch.tensor([[True] * 5, [True, True, True, False, False]])
    visited = torch.tensor([[False, True, True, False, False],
                            [False, True, False, False, False]])
    base = torch.randn(2, 5)

    residual, coverage, marginal = module(evidence, valid, visited)
    assert sum(parameter.numel() for parameter in module.parameters()) == 449
    assert torch.count_nonzero(residual) == 0
    assert torch.equal(base + residual, base)
    assert torch.count_nonzero(residual[:, 0]) == 0
    assert torch.count_nonzero(residual[0, 1:3]) == 0
    assert torch.count_nonzero(residual[1, 3:]) == 0
    assert torch.isfinite(coverage).all() and torch.isfinite(marginal).all()

    residual.sum().backward()
    assert evidence.grad is None
    assert module.output.weight.grad.abs().sum() > 0


def test_graph_memory_merges_instruction_evidence_with_idempotent_max(
        graph_utils):
    graph = graph_utils.GraphMap(
        has_real_pos=False,
        loc_noise=0.5,
        merge_ghost=True,
        ghost_aug=0.0,
        instruction_evidence_size=4,
    )
    common = dict(
        prev_vp=None,
        step_id=1,
        cur_vp='0',
        cur_pos=np.array([0.0, 0.0, 0.0]),
        cur_embeds=torch.ones(3),
        cand_vp=['0_0'],
        cand_pos=[np.array([1.0, 0.0, 0.0])],
        cand_embeds=torch.ones(1, 3),
        cand_real_pos=None,
        cur_instruction_evidence=torch.tensor([0.2, 0.4, 0.1, 0.0]),
        cand_instruction_evidence=torch.tensor([[0.3, 0.1, 0.5, 0.0]]),
    )
    targets = graph.update_graph(**common)
    ghost = targets[0]
    common.update(
        prev_vp='0',
        step_id=2,
        cur_vp='1',
        cur_pos=np.array([0.0, 0.0, -1.0]),
        cand_vp=['1_0'],
        cand_pos=[np.array([1.1, 0.0, 0.0])],
        cur_instruction_evidence=torch.tensor([0.4, 0.2, 0.1, 0.1]),
        cand_instruction_evidence=torch.tensor([[0.2, 0.8, 0.5, 0.0]]),
    )
    graph.update_graph(**common)

    assert torch.equal(
        graph.get_instruction_evidence(ghost),
        torch.tensor([0.3, 0.8, 0.5, 0.0]),
    )
    assert torch.equal(
        graph.get_instruction_evidence('1'),
        torch.tensor([0.4, 0.2, 0.1, 0.1]),
    )
    graph.delete_ghost(ghost)
    assert ghost not in graph.ghost_instruction_evidence


def test_graph_memory_requires_complete_evidence_when_enabled(graph_utils):
    graph = graph_utils.GraphMap(
        False, 0.5, True, 0.0, instruction_evidence_size=4
    )
    with pytest.raises(ValueError, match='missing or misaligned'):
        graph.update_graph(
            None, 1, '0', np.zeros(3), torch.zeros(3),
            [], [], torch.empty(0, 3), None,
        )
