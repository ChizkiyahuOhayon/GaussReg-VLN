"""Exercise E19 semantic geometry through the production navigation model."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model(monkeypatch):
    package = ModuleType('vlnce_baselines')
    package.__path__ = [str(ROOT / 'vlnce_baselines')]
    common = ModuleType('vlnce_baselines.common')
    common.__path__ = [str(ROOT / 'vlnce_baselines/common')]
    transformers = ModuleType('transformers')

    class Constructor(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def init_weights(self):
            pass

    transformers.BertPreTrainedModel = Constructor
    for name, module in [
            ('vlnce_baselines', package),
            ('vlnce_baselines.common', common),
            ('transformers', transformers)]:
        monkeypatch.setitem(sys.modules, name, module)
    load(monkeypatch, 'vlnce_baselines.common.transformer',
         'vlnce_baselines/common/transformer.py')
    load(monkeypatch, 'vlnce_baselines.common.ops',
         'vlnce_baselines/common/ops.py')
    load(monkeypatch, 'vlnce_baselines.geo_token',
         'vlnce_baselines/geo_token.py')
    load(monkeypatch, 'vlnce_baselines.landmark_transport',
         'vlnce_baselines/landmark_transport.py')
    load(monkeypatch, 'vlnce_baselines.factorized_landmark',
         'vlnce_baselines/factorized_landmark.py')
    load(monkeypatch, 'vlnce_baselines.transient_local_geometry',
         'vlnce_baselines/transient_local_geometry.py')
    module = load(monkeypatch, 'e19_real_forward',
                  'vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py')
    config = SimpleNamespace(
        hidden_size=16, vocab_size=32, max_position_embeddings=32,
        type_vocab_size=2, max_txt_task_embeddings=4, layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0, num_attention_heads=4,
        output_attentions=True, attention_probs_dropout_prob=0.0,
        intermediate_size=32, hidden_act='gelu', num_l_layers=1,
        update_lang_bert=False, image_feat_size=8, use_depth_embedding=True,
        depth_feat_size=4, angle_feat_size=4, num_pano_layers=1,
        max_action_steps=100, max_gmap_task_embeddings=3, num_x_layers=2,
        use_lang2visn_attn=True, graph_sprels=True,
        pred_head_dropout_prob=0.0, fix_lang_embedding=True,
        fix_pano_embedding=True, instruction_coverage_hidden_size=0,
        successor_hidden_size=0, landmark_transport_size=0,
        factorized_landmark_size=8, transient_geometry_size=8,
    )
    torch.manual_seed(19)
    result = module.GlocalTextPathNavCMT(config)
    result.eval()
    return result


def navigation_inputs(width=16):
    return dict(
        txt_embeds=torch.randn(2, 8, width),
        txt_masks=torch.tensor([
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False, False],
        ]),
        gmap_vpids=[[None, '0', '1', 'g0', 'g1']] * 2,
        gmap_step_ids=torch.tensor([[0, 1, 2, 0, 0]] * 2),
        gmap_img_fts=torch.randn(2, 5, width),
        gmap_pos_fts=torch.randn(2, 5, 7),
        gmap_masks=torch.tensor([
            [True, True, True, True, True],
            [True, True, True, True, False],
        ]),
        gmap_visited_masks=torch.tensor([
            [False, True, True, False, False],
            [False, True, False, False, False],
        ]),
        gmap_pair_dists=torch.zeros(2, 5, 5),
        gmap_task_embeddings=torch.ones(2, 5, dtype=torch.long),
        gmap_transport_views=torch.randn(2, 5, 3, width),
        gmap_transport_masks=torch.tensor([
            [[False] * 3, [True] * 3, [True, False, False],
             [True, True, False], [True, False, False]],
            [[False] * 3, [True, True, False], [True, False, False],
             [True, False, False], [False] * 3],
        ]),
        gmap_local_geometry=torch.randn(2, 5, 16, 3),
        gmap_local_geometry_masks=torch.tensor([
            [False, False, False, True, True],
            [False, False, False, True, False],
        ]),
    )


def test_zero_geometry_exactly_recovers_e15(model):
    inputs = navigation_inputs()
    enabled = model.forward_navigation(**inputs)
    geometry = model.transient_geometry
    model.transient_geometry = None
    disabled = model.forward_navigation(**{
        key: value for key, value in inputs.items()
        if not key.startswith('gmap_local_geometry')
    })
    model.transient_geometry = geometry

    finite = torch.isfinite(disabled['global_logits'])
    assert torch.equal(enabled['global_logits'][finite],
                       disabled['global_logits'][finite])
    assert torch.equal(torch.isneginf(enabled['global_logits']),
                       torch.isneginf(disabled['global_logits']))
    assert enabled['geometry_residual_norm'].item() == 0.0


def test_navigation_gradient_reaches_both_e19_branches_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.factorized_landmark, model.transient_geometry):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        module.output.weight.data.normal_(std=0.01)

    outputs = model.forward_navigation(**navigation_inputs())
    base_stop = torch.softmax(outputs['base_global_logits'], dim=-1)[:, 0]
    assert torch.allclose(outputs['global_logits'].exp()[:, 0], base_stop)
    outputs['global_logits'][:, 3].sum().backward()

    for module in (model.factorized_landmark, model.transient_geometry):
        assert all(parameter.grad is not None for parameter in module.parameters())
        assert all(torch.isfinite(parameter.grad).all()
                   for parameter in module.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not (name.startswith('factorized_landmark.') or
                name.startswith('transient_geometry.'))
    )
