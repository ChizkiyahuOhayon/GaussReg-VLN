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
