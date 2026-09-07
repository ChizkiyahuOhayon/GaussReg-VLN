import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'successor_for_test', ROOT / 'vlnce_baselines/successor.py'
)
SEG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEG)


def test_successor_decoder_uses_language_and_ignores_padding():
    torch.manual_seed(7)
    decoder = SEG.SuccessorDecoder(16, 8)
    graph = torch.randn(2, 4, 16, requires_grad=True)
    text = torch.randn(2, 3, 16, requires_grad=True)
    gm = torch.tensor([[True, True, True, False]] * 2)
    tm = torch.tensor([[True, True, False]] * 2)
    prediction = decoder(graph, gm, text, tm)
    padded_graph, padded_text = graph.detach().clone(), text.detach().clone()
    padded_graph[:, -1] = 1000
    padded_text[:, -1] = -1000
    other = decoder(padded_graph, gm, padded_text, tm)
    assert prediction.shape == graph.shape
    assert torch.allclose(prediction[:, :3], other[:, :3], atol=1e-6)
    changed_text = padded_text.clone()
    changed_text[:, :2] *= -1
    assert not torch.allclose(other[:, :3], decoder(
        padded_graph, gm, changed_text, tm
    )[:, :3])
    prediction[:, :3].square().sum().backward()
    assert graph.grad is None and text.grad is None
    assert decoder.output.weight.grad.abs().sum() > 0
    assert decoder.attention.in_proj_weight.grad.abs().sum() > 0


def test_virtual_graph_preserves_owners_and_only_exposes_frontier_evidence():
    graph = torch.arange(4 * 8).reshape(1, 4, 8).float()
    future = graph + 100
    position = torch.ones_like(graph)
    valid = torch.tensor([[True, True, True, False]])
    visited = torch.tensor([[False, True, False, False]])
    dists = torch.tensor([[[0., 0., 0., 0.], [0., 0., 2., 0.],
                           [0., 2., 0., 0.], [0., 0., 0., 0.]]])
    features, masks, distances = SEG.expand_successor_graph(
        graph, future, position, valid, visited, dists
    )
    assert torch.equal(features[:, :4], graph)
    assert torch.equal(features[:, 6], future[:, 2] + position[:, 2])
    assert masks.tolist() == [[True, True, True, False, False, False, True, False]]
    assert distances[0, 1, 6] == dists[0, 1, 2]
    assert distances[0, 2, 6] == 0
    assert torch.equal(distances[:, :4, :4], dists)


def _step(indices, actions, positions, features, move=None):
    n = len(indices)
    return {
        'indices': indices, 'action': torch.tensor(actions),
        'executed_move': torch.tensor(move if move is not None else [True] * n),
        'current_position': torch.tensor(positions).float(),
        'successor_position': torch.tensor([[1., 0., 0.]] * n),
        'input': {
            'gmap_img_fts': torch.tensor(features).float().reshape(n, 3, 2),
            'gmap_step_ids': torch.tensor([[0, 2, 0]] * n),
            'gmap_masks': torch.ones(n, 3, dtype=torch.bool),
            'gmap_visited_masks': torch.tensor([[False, True, False]] * n),
        },
    }


def test_arrivals_follow_environment_identity_after_pause_not_batch_row():
    a = _step([0, 1], [0, 2], [[0, 0, 0]] * 2, list(range(12)), [False, True])
    b = _step([1], [0], [[1, 0, 0]], [0, 0, 11, 12, 0, 0], [False])
    pairs = SEG.arrival_pairs([a, b])
    assert len(pairs) == 1
    step, row, action, target = pairs[0]
    assert (step, row, action) == (0, 1, 2)
    assert target.tolist() == [11., 12.]


def test_stop_forced_termination_and_failed_arrival_have_no_targets():
    a = _step([0, 1, 2], [0, 2, 2], [[0, 0, 0]] * 3,
              list(range(18)), [False, False, True])
    b = _step([0, 1, 2], [0, 0, 0], [[1, 0, 0], [1, 0, 0], [3, 0, 0]],
              list(range(18)), [False] * 3)
    assert SEG.arrival_pairs([a, b]) == []
    assert SEG.arrival_pairs([a]) == []


def test_loss_matches_categorical_kl_and_does_not_backpropagate_to_targets():
    prediction = torch.zeros(1, 3, requires_grad=True)
    arrival = torch.ones(1, 3, requires_grad=True)
    student = torch.tensor([[0., -float('inf'), 1.]], requires_grad=True)
    teacher = torch.tensor([[1., -float('inf'), 0.]], requires_grad=True)
    feature, planner = SEG.successor_loss(prediction, arrival, student, teacher)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=teacher[:, [0, 2]]),
        torch.distributions.Categorical(logits=student[:, [0, 2]]),
    ).mean()
    assert torch.allclose(planner, expected)
    (feature + planner).backward()
    assert torch.isfinite(student.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert arrival.grad is None and teacher.grad is None
    assert student.grad[0, 1] == 0


def test_single_stop_action_has_finite_zero_planner_loss():
    logits = torch.tensor([[0., -float('inf')]], requires_grad=True)
    feature, planner = SEG.successor_loss(
        torch.ones(1, 2), torch.ones(1, 2), logits, logits.detach()
    )
    assert feature == 0 and planner == 0
    planner.backward()
    assert torch.isfinite(logits.grad).all()


def test_nonfinite_training_evidence_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        SEG.successor_loss(torch.tensor([[float('nan')]]), torch.ones(1, 1),
                           torch.zeros(1, 2), torch.zeros(1, 2))
