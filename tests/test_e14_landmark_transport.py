import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'landmark_transport_for_test',
    ROOT / 'vlnce_baselines/landmark_transport.py',
)
LTG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LTG)


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
            'graph_utils_for_e14',
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


def test_ordered_slot_content_partitions_tokens_and_ignores_padding():
    embeddings = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
    masks = torch.tensor([
        [True, True, True, True, True, True, True, True],
        [True, True, False, False, False, False, False, False],
    ])
    slots, slot_masks = LTG.ordered_slot_content(embeddings, masks)

    assert slots.shape == (2, 4, 4)
    assert torch.equal(slots[0, 0], embeddings[0, :2].mean(0))
    assert torch.equal(slots[0, 3], embeddings[0, 6:8].mean(0))
    assert torch.equal(slots[1, 0], embeddings[1, 0])
    assert torch.equal(slots[1, 2], embeddings[1, 1])
    assert slot_masks.tolist() == [[True] * 4, [True, False, True, False]]


def test_transport_is_padding_invariant_and_zero_initialized_to_e0():
    torch.manual_seed(14)
    module = LTG.LandmarkTransport(16, 8)
    text = torch.randn(2, 8, 16)
    text_masks = torch.tensor([
        [True, True, True, True, True, True, False, False],
        [True, True, True, True, False, False, False, False],
    ])
    views = torch.randn(2, 5, 3, 16)
    view_masks = torch.tensor([
        [[False, False, False], [True, True, True], [True, False, False],
         [True, True, False], [False, False, False]],
        [[False, False, False], [True, True, False], [True, False, False],
         [False, False, False], [False, False, False]],
    ])
    base = torch.randn(2, 5, 16)

    residual, diagnostics = module(views, view_masks, text, text_masks)
    changed = views.clone()
    changed[view_masks.logical_not()] = 10000.0
    other, _ = module(changed, view_masks, text, text_masks)

    assert sum(parameter.numel() for parameter in module.parameters()) == 800
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(base + residual, base)
    assert torch.equal(residual, other)
    assert torch.count_nonzero(residual[:, 0]) == 0
    assert torch.isfinite(diagnostics['entropy'])
    assert diagnostics['residual_norm'].item() == 0.0


def test_navigation_gradient_reaches_all_transport_projections_after_warmup():
    torch.manual_seed(14)
    module = LTG.LandmarkTransport(16, 8)
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)
    views = torch.randn(2, 4, 3, 16)
    masks = torch.ones(2, 4, 3, dtype=torch.bool)
    masks[:, 0] = False
    text = torch.randn(2, 8, 16)
    text_masks = torch.ones(2, 8, dtype=torch.bool)

    module(views, masks, text, text_masks)[0].sum().backward()

    assert module.view_projection.weight.grad.abs().sum() > 0
    assert module.text_projection.weight.grad.abs().sum() > 0
    assert module.output.weight.grad.abs().sum() > 0
    assert views.grad is None and text.grad is None


def test_transport_is_source_permutation_invariant_and_instruction_ordered():
    torch.manual_seed(14)
    module = LTG.LandmarkTransport(16, 8)
    with torch.no_grad():
        module.output.weight.normal_(std=0.01)
    views = torch.randn(1, 2, 3, 16)
    view_masks = torch.tensor([[[True, True, True], [True, True, False]]])
    text = torch.randn(1, 8, 16)
    text_masks = torch.ones(1, 8, dtype=torch.bool)

    output, _ = module(views, view_masks, text, text_masks)
    permutation = torch.tensor([2, 0, 1])
    permuted, _ = module(
        views[:, :, permutation], view_masks[:, :, permutation],
        text, text_masks,
    )
    reversed_text, _ = module(
        views, view_masks, text.flip(1), text_masks,
    )

    assert torch.allclose(output, permuted, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(output, reversed_text)


def test_graph_memory_preserves_prepool_sources_and_deletes_ghost(graph_utils):
    graph = graph_utils.GraphMap(
        False, 0.5, True, 0.0, transport_memory=True
    )
    current_views = torch.randn(3, 6, requires_grad=True)
    candidate_views = torch.randn(1, 6, requires_grad=True)
    targets = graph.update_graph(
        None, 1, '0', np.zeros(3), torch.zeros(6), ['0_0'],
        [np.array([1.0, 0.0, 0.0])], torch.zeros(1, 6), None,
        cur_transport_views=current_views,
        cand_transport_views=candidate_views,
    )
    ghost = targets[0]

    assert torch.equal(graph.get_transport_views('0'), current_views.detach())
    assert torch.equal(graph.get_transport_views(ghost), candidate_views.detach())
    assert graph.get_transport_views('0').device.type == 'cpu'
    assert not graph.get_transport_views('0').requires_grad
    graph.delete_ghost(ghost)
    assert ghost not in graph.ghost_transport_views


def test_graph_memory_requires_aligned_sources_when_enabled(graph_utils):
    graph = graph_utils.GraphMap(
        False, 0.5, True, 0.0, transport_memory=True
    )
    with pytest.raises(ValueError, match='Transport views are missing or misaligned'):
        graph.update_graph(
            None, 1, '0', np.zeros(3), torch.zeros(6), [], [],
            torch.empty(0, 6), None,
        )
