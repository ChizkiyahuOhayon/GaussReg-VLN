import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def graph_utils():
    habitat = types.ModuleType('habitat')
    habitat_tasks = types.ModuleType('habitat.tasks')
    habitat_task_utils = types.ModuleType('habitat.tasks.utils')
    habitat_task_utils.cartesian_to_polar = lambda x, y: (
        np.hypot(x, y), np.arctan2(y, x)
    )
    habitat_utils = types.ModuleType('habitat.utils')
    habitat_geometry = types.ModuleType('habitat.utils.geometry_utils')
    habitat_geometry.quaternion_rotate_vector = lambda quat, vector: vector
    habitat_geometry.quaternion_from_coeff = lambda coeff: coeff

    modules = {
        'habitat': habitat,
        'habitat.tasks': habitat_tasks,
        'habitat.tasks.utils': habitat_task_utils,
        'habitat.utils': habitat_utils,
        'habitat.utils.geometry_utils': habitat_geometry,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module = _load_module(
            'etp_graph_utils_for_test',
            REPO_ROOT / 'vlnce_baselines/models/graph_utils.py',
        )
        module.heading_from_quaternion = lambda quaternion: 0.0
        yield module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


@pytest.fixture(scope='module')
def vilmodel():
    transformers = types.ModuleType('transformers')

    class BertPreTrainedModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    transformers.BertPreTrainedModel = BertPreTrainedModel
    package = types.ModuleType('vlnce_baselines')
    common = types.ModuleType('vlnce_baselines.common')
    ops = types.ModuleType('vlnce_baselines.common.ops')
    ops.create_transformer_encoder = lambda *args, **kwargs: nn.Identity()
    ops.extend_neg_masks = lambda masks: masks
    ops.gen_seq_masks = lambda lengths: lengths
    ops.pad_tensors_wgrad = lambda tensors: tensors

    modules = {
        'transformers': transformers,
        'vlnce_baselines': package,
        'vlnce_baselines.common': common,
        'vlnce_baselines.common.ops': ops,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield _load_module(
            'etp_vilmodel_for_test',
            REPO_ROOT / 'vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py',
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _graph_map(module, gauss_feat_size):
    graph = module.GraphMap(
        has_real_pos=False,
        loc_noise=0.5,
        merge_ghost=True,
        ghost_aug=0.0,
        gauss_feat_size=gauss_feat_size,
    )
    graph.node_pos = {
        '0': np.array([0., 0., 0.], dtype=np.float32),
        '1': np.array([0., 0., -1.], dtype=np.float32),
    }
    graph.ghost_pos = {
        'g0': [
            np.array([1., 0., 0.], dtype=np.float32),
            np.array([3., 0., 0.], dtype=np.float32),
        ]
    }
    graph.ghost_aug_pos = {
        'g0': np.array([2., 0., 0.], dtype=np.float32)
    }
    graph.ghost_fronts = {'g0': ['0', '1']}
    graph.shortest_dist = {'0': {'0': 0., '1': 1.}}
    graph.shortest_path = {'0': {'0': ['0'], '1': ['0', '1']}}
    return graph


def _encoder_config(gauss_feat_size, gauss_residual_scale=1.0):
    return SimpleNamespace(
        angle_feat_size=4,
        gauss_feat_size=gauss_feat_size,
        gauss_residual_scale=gauss_residual_scale,
        hidden_size=768,
        layer_norm_eps=1e-12,
        max_action_steps=100,
        max_gmap_task_embeddings=3,
        hidden_dropout_prob=0.0,
        graph_sprels=False,
    )


def _new_encoder(module, gauss_feat_size, gauss_residual_scale=1.0):
    original_encoder = module.CrossmodalEncoder
    module.CrossmodalEncoder = lambda config: nn.Identity()
    try:
        return module.GlobalMapEncoder(
            _encoder_config(gauss_feat_size, gauss_residual_scale)
        )
    finally:
        module.CrossmodalEncoder = original_encoder


def test_graph_map_preserves_baseline_features_and_appends_gaussian_state(graph_utils):
    baseline = _graph_map(graph_utils, gauss_feat_size=0)
    gaussian = _graph_map(graph_utils, gauss_feat_size=5)
    viewpoint_ids = [None, '1', 'g0']

    baseline_features = baseline.get_pos_fts(
        '0', baseline.node_pos['0'], np.zeros(4), viewpoint_ids
    )
    gaussian_features = gaussian.get_pos_fts(
        '0', gaussian.node_pos['0'], np.zeros(4), viewpoint_ids
    )

    assert baseline_features.shape == (3, 7)
    assert gaussian_features.shape == (3, 12)
    np.testing.assert_array_equal(gaussian_features[:, :7], baseline_features)
    np.testing.assert_allclose(
        gaussian_features[:, 7:],
        np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0.125, 0.25],
            [2., 0., 0., 0.25, 0.5],
        ], dtype=np.float32),
    )


def test_zero_initialized_residual_is_an_exact_no_op(vilmodel):
    encoder = _new_encoder(vilmodel, gauss_feat_size=5)
    position_features = torch.randn(2, 4, 12)

    expected = encoder.gmap_pos_embeddings(position_features[..., :7])
    actual = encoder.position_embedding(position_features)

    assert torch.count_nonzero(encoder.gmap_gauss_embedding.weight) == 0
    assert torch.equal(actual, expected)

    actual.sum().backward()
    assert torch.count_nonzero(encoder.gmap_gauss_embedding.weight.grad) > 0


def test_gaussian_residual_scale_preserves_default_and_scales_residual(vilmodel):
    default = _new_encoder(vilmodel, gauss_feat_size=5)
    disabled = _new_encoder(
        vilmodel, gauss_feat_size=5, gauss_residual_scale=0.0
    )
    scaled = _new_encoder(
        vilmodel, gauss_feat_size=5, gauss_residual_scale=0.25
    )
    disabled.load_state_dict(default.state_dict())
    scaled.load_state_dict(default.state_dict())
    position_features = torch.randn(2, 4, 12)

    with torch.no_grad():
        default.gmap_gauss_embedding.weight.fill_(0.01)
        disabled.gmap_gauss_embedding.weight.fill_(0.01)
        scaled.gmap_gauss_embedding.weight.fill_(0.01)

    base = default.gmap_pos_embeddings(position_features[..., :7])
    residual = default.gmap_gauss_embedding(position_features[..., 7:])

    assert torch.equal(
        default.position_embedding(position_features), base + residual
    )
    assert torch.equal(
        scaled.position_embedding(position_features), base + 0.25 * residual
    )
    assert torch.equal(disabled.position_embedding(position_features), base)


def test_old_state_dict_leaves_only_zero_gaussian_weight_missing(vilmodel):
    baseline = _new_encoder(vilmodel, gauss_feat_size=0)
    gaussian = _new_encoder(vilmodel, gauss_feat_size=5)

    incompatible = gaussian.load_state_dict(baseline.state_dict(), strict=False)

    assert incompatible.missing_keys == ['gmap_gauss_embedding.weight']
    assert incompatible.unexpected_keys == []
    assert torch.count_nonzero(gaussian.gmap_gauss_embedding.weight) == 0


def test_gaussian_only_mode_has_one_trainable_3840_parameter_tensor(vilmodel):
    encoder = _new_encoder(vilmodel, gauss_feat_size=5)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    for parameter in encoder.gmap_gauss_embedding.parameters():
        parameter.requires_grad = True

    trainable = [parameter for parameter in encoder.parameters()
                 if parameter.requires_grad]

    assert len(trainable) == 1
    assert trainable[0].numel() == 3840


def test_position_feature_shape_mismatch_fails_fast(vilmodel):
    encoder = _new_encoder(vilmodel, gauss_feat_size=5)

    with pytest.raises(ValueError, match='Expected 12 graph position features'):
        encoder.position_embedding(torch.zeros(1, 3, 7))


def _new_candidate_scorer(module, scale=1.0):
    config = SimpleNamespace(
        hidden_size=768,
        layer_norm_eps=1e-12,
        candidate_scorer_scale=scale,
    )
    scorer = module.CandidateResidualScorer(
        config, position_size=12, hidden_size=256
    )
    scorer.reset_output()
    return scorer


def test_candidate_scorer_is_zero_initialized_lightweight_and_detached(vilmodel):
    scorer = _new_candidate_scorer(vilmodel)
    representations = torch.randn(2, 4, 1536, requires_grad=True)
    positions = torch.randn(2, 4, 12)
    visited = torch.tensor([
        [False, True, False, False],
        [False, False, True, False],
    ])
    baseline_logits = torch.randn(2, 4)

    residual = scorer(representations, positions, visited)

    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(baseline_logits + residual, baseline_logits)
    assert sum(parameter.numel() for parameter in scorer.parameters()) == 400385

    residual.sum().backward()
    assert representations.grad is None
    assert torch.count_nonzero(scorer.output.weight.grad) > 0


def test_candidate_scorer_has_an_explicit_stop_indicator(vilmodel):
    scorer = _new_candidate_scorer(vilmodel)
    with torch.no_grad():
        scorer.hidden.weight.zero_()
        scorer.hidden.bias.zero_()
        scorer.hidden.weight[0, -1] = 1.0
        scorer.output.weight.zero_()
        scorer.output.bias.zero_()
        scorer.output.weight[0, 0] = 1.0

    representations = torch.zeros(1, 3, 1536)
    positions = torch.zeros(1, 3, 12)
    visited = torch.zeros(1, 3, dtype=torch.bool)
    residual = scorer(representations, positions, visited)

    assert residual[0, 0] > 0
    assert torch.equal(residual[0, 1:], torch.zeros(2))


def test_candidate_scorer_scale_zero_restores_baseline(vilmodel):
    scorer = _new_candidate_scorer(vilmodel, scale=0.0)
    with torch.no_grad():
        scorer.output.weight.fill_(1.0)
        scorer.output.bias.fill_(1.0)

    residual = scorer(
        torch.randn(1, 3, 1536),
        torch.randn(1, 3, 12),
        torch.zeros(1, 3, dtype=torch.bool),
    )

    assert torch.equal(residual, torch.zeros_like(residual))


def test_candidate_scorer_rejects_misaligned_inputs(vilmodel):
    scorer = _new_candidate_scorer(vilmodel)
    representations = torch.zeros(1, 3, 1536)
    visited = torch.zeros(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match='Expected 12 candidate position features'):
        scorer(representations, torch.zeros(1, 3, 7), visited)
    with pytest.raises(ValueError, match='visit mask are misaligned'):
        scorer(representations, torch.zeros(1, 3, 12), visited[:, :2])


def _new_gaussian_bev(module):
    field = module.GaussianBEVResidual(
        representation_size=1536,
        position_size=12,
        hidden_size=32,
        grid_size=21,
        extent=10.0,
        max_distance=30.0,
        location_noise=0.5,
    )
    field.reset_output()
    return field


def _gaussian_bev_inputs(batch_size=2):
    representations = torch.randn(batch_size, 4, 1536, requires_grad=True)
    positions = torch.zeros(batch_size, 4, 12)
    positions[:, :, 1] = 1.0
    positions[:, :, 3] = 1.0
    positions[:, 1:, 4] = torch.tensor([0.1, 0.2, 0.3])
    positions[:, 1:, 7] = 0.2
    positions[:, 1:, 9] = 0.4
    positions[:, 1:, 10] = 0.5
    masks = torch.ones(batch_size, 4, dtype=torch.bool)
    visited = torch.zeros(batch_size, 4, dtype=torch.bool)
    visited[:, 1] = True
    return representations, positions, masks, visited


def test_gaussian_bev_is_zero_initialized_lightweight_and_detached(vilmodel):
    field = _new_gaussian_bev(vilmodel)
    representations, positions, masks, visited = _gaussian_bev_inputs()
    baseline_logits = torch.randn(2, 4)

    residual = field(representations, positions, masks, visited)

    assert residual.shape == baseline_logits.shape
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(baseline_logits + residual, baseline_logits)
    assert sum(parameter.numel() for parameter in field.parameters()) < 100000

    residual.sum().backward()
    assert representations.grad is None
    assert torch.count_nonzero(field.output.weight.grad) > 0


def test_gaussian_bev_excludes_stop_padding_and_out_of_range_tokens(vilmodel):
    field = _new_gaussian_bev(vilmodel)
    _, positions, masks, _ = _gaussian_bev_inputs(batch_size=1)
    masks[0, 3] = False
    positions[0, 2, 4] = 1.0

    weights, spatial_mask = field._gaussian_weights(positions, masks)

    assert torch.count_nonzero(weights[:, 0]) == 0
    assert torch.count_nonzero(weights[:, 2]) == 0
    assert torch.count_nonzero(weights[:, 3]) == 0
    assert not spatial_mask[0, 0]
    assert not spatial_mask[0, 2]
    assert not spatial_mask[0, 3]


def test_gaussian_bev_padding_and_batches_are_isolated(vilmodel):
    torch.manual_seed(0)
    field = _new_gaussian_bev(vilmodel)
    with torch.no_grad():
        field.output.weight.fill_(1.0)
    representations, positions, masks, visited = _gaussian_bev_inputs()
    masks[:, 3] = False

    expected = field(representations, positions, masks, visited)
    changed = representations.detach().clone()
    changed[0, 3].fill_(1e4)
    changed[1, 3].fill_(-1e4)
    actual = field(changed, positions, masks, visited)

    assert torch.equal(actual, expected)
    assert torch.allclose(actual[0], field(
        changed[:1], positions[:1], masks[:1], visited[:1]
    )[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual[1], field(
        changed[1:], positions[1:], masks[1:], visited[1:]
    )[0], atol=1e-6, rtol=1e-6)


def test_gaussian_bev_uncertainty_controls_spatial_support(vilmodel):
    field = _new_gaussian_bev(vilmodel)
    _, positions, masks, _ = _gaussian_bev_inputs(batch_size=1)
    masks[:, 2:] = False
    positions[0, 1, 4] = 0.0

    positions[0, 1, 7] = 0.0
    positions[0, 1, 9] = 0.0
    narrow, _ = field._gaussian_weights(positions, masks)
    positions[0, 1, 7] = 3.0
    positions[0, 1, 9] = 3.0
    wide, _ = field._gaussian_weights(positions, masks)

    center = field.grid_size // 2
    assert wide[0, 1, center, center + 2] > narrow[0, 1, center, center + 2]
    assert wide[0, 1].sum() > narrow[0, 1].sum()


def _new_anchor_repair(module):
    repair = module.AnchorRelativeRepair(
        representation_size=1536,
        position_size=12,
        hidden_size=64,
        layer_norm_eps=1e-12,
    )
    repair.reset_output()
    return repair


def test_anchor_repair_is_an_exact_greedy_no_op_and_detached(vilmodel):
    repair = _new_anchor_repair(vilmodel)
    representations, positions, masks, visited = _gaussian_bev_inputs()
    base_logits = torch.tensor([
        [0.2, 10.0, 1.0, 0.5],
        [1.5, 10.0, 1.0, 0.5],
    ])

    repair_logits = repair(
        base_logits, representations, positions, masks, visited
    )
    masked_base = base_logits.masked_fill(visited, -float('inf'))

    assert torch.equal(repair_logits.argmax(dim=-1), masked_base.argmax(dim=-1))
    assert torch.isneginf(repair_logits[visited]).all()
    assert sum(parameter.numel() for parameter in repair.parameters()) == 154972
    assert torch.count_nonzero(repair.output.weight) == 0

    repair_logits[torch.isfinite(repair_logits)].sum().backward()
    assert representations.grad is None
    assert torch.count_nonzero(repair.output.weight.grad) > 0


def test_anchor_repair_strictly_prefers_keep_on_tied_base_logits(vilmodel):
    repair = _new_anchor_repair(vilmodel)
    representations, positions, masks, visited = _gaussian_bev_inputs(
        batch_size=1
    )
    masks[:, 2:] = False
    visited.zero_()
    base_logits = torch.zeros(1, 4)

    repair_logits = repair(
        base_logits, representations, positions, masks, visited
    )

    assert repair_logits.argmax(dim=-1).item() == 0
    assert repair_logits[0, 0].item() == 0.0
    assert repair_logits[0, 1].item() < 0.0
    assert torch.isneginf(repair_logits[0, 2:]).all()


def test_anchor_relative_advantages_use_the_base_trajectory():
    module = _load_module(
        'anchor_relative_for_test',
        REPO_ROOT / 'vlnce_baselines/anchor_relative.py',
    )
    rewards = [
        [2.0, None],
        [3.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [2.0, 4.0],
        [2.0, 5.0],
        [2.0, 6.0],
        [2.0, 7.0],
    ]

    advantages = module.anchor_relative_advantages(rewards)

    assert advantages[0] == [0.0, 0.0]
    assert advantages[1][0] == pytest.approx(np.sqrt(7.0 / 2.0))
    assert advantages[2][0] == pytest.approx(-np.sqrt(7.0 / 2.0))
    assert advantages[3][0] == 0.0
    assert all(row[1] == 0.0 for row in advantages)

    equal_rewards = [[2.0], [2.0], [2.0], [2.0]]
    assert module.anchor_relative_advantages(equal_rewards) == [
        [0.0], [0.0], [0.0], [0.0]
    ]


def test_gaussian_bev_out_of_range_candidate_falls_back_to_e0(vilmodel):
    field = _new_gaussian_bev(vilmodel)
    with torch.no_grad():
        field.output.weight.fill_(1.0)
    representations, positions, masks, visited = _gaussian_bev_inputs(
        batch_size=1
    )
    positions[0, 2, 4] = 1.0

    residual = field(representations, positions, masks, visited)

    assert residual.dtype == representations.dtype
    assert residual[0, 2].item() == 0.0
