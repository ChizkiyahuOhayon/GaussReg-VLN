"""Exercise E16 budgeted routing through the real navigation model."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def test_default_config_registers_e16_keys():
    source = (ROOT / 'vlnce_baselines/config/default.py').read_text()
    assert '_C.GRPO.budgeted_factorized_landmark_only = False' in source
    assert '_C.MODEL.factorized_landmark_budgeted = False' in source


def test_vlnbert_loader_propagates_e16_mode():
    source = (
        ROOT / 'vlnce_baselines/models/etp/ETP_R1_vlnbert_init.py'
    ).read_text()
    assert 'vis_config.factorized_landmark_budgeted = getattr(' in source


@pytest.fixture
def model(monkeypatch):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

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
    load('vlnce_baselines.common.transformer',
         'vlnce_baselines/common/transformer.py')
    load('vlnce_baselines.common.ops', 'vlnce_baselines/common/ops.py')
    load('vlnce_baselines.geo_token', 'vlnce_baselines/geo_token.py')
    load('vlnce_baselines.landmark_transport',
         'vlnce_baselines/landmark_transport.py')
    load('vlnce_baselines.factorized_landmark',
         'vlnce_baselines/factorized_landmark.py')
    load('vlnce_baselines.budgeted_factorized_landmark',
         'vlnce_baselines/budgeted_factorized_landmark.py')
    module = load('e16_real_forward',
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
        factorized_landmark_size=8, factorized_landmark_budgeted=True,
    )
    torch.manual_seed(16)
    result = module.GlocalTextPathNavCMT(config)
    result.eval()
    return result


def navigation_inputs(width=16):
    view_masks = torch.tensor([[
        [False, False, False], [True, True, True], [True, False, False],
        [True, True, False], [True, False, False],
    ]])
    pair_dists = torch.zeros(1, 5, 5)
    return dict(
        txt_embeds=torch.randn(1, 8, width),
        txt_masks=torch.tensor([[
            True, True, True, True, True, True, False, False,
        ]]),
        gmap_vpids=[[None, '0', '1', 'g0', 'g1']],
        gmap_step_ids=torch.tensor([[0, 1, 2, 0, 0]]),
        gmap_img_fts=torch.randn(1, 5, width),
        gmap_pos_fts=torch.randn(1, 5, 7),
        gmap_masks=torch.tensor([[True, True, True, True, True]]),
        gmap_visited_masks=torch.tensor([[False, True, True, False, False]]),
        gmap_pair_dists=pair_dists,
        gmap_task_embeddings=torch.ones(1, 5, dtype=torch.long),
        gmap_transport_views=torch.randn(1, 5, 3, width),
        gmap_transport_masks=view_masks,
    )


def test_zero_landmark_is_exact_e0_with_inactive_budget(model):
    inputs = navigation_inputs()
    enabled = model.forward_navigation(**inputs)
    transport = model.factorized_landmark
    model.factorized_landmark = None
    disabled = model.forward_navigation(**{
        key: value for key, value in inputs.items()
        if not key.startswith('gmap_transport_')
    })
    model.factorized_landmark = transport

    expected = torch.log_softmax(disabled['global_logits'], dim=-1)
    finite = torch.isfinite(expected)
    assert torch.allclose(enabled['global_logits'][finite], expected[finite])
    assert torch.equal(torch.isneginf(enabled['global_logits']),
                       torch.isneginf(expected))
    assert enabled['budget_activation_rate'].item() == 0.0
    assert torch.equal(enabled['factorized_greedy_actions'],
                       disabled['global_logits'].argmax(-1))


def test_budget_constrains_rerouting_and_only_trains_landmark(model):
    inputs = navigation_inputs()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.factorized_landmark.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        model.factorized_landmark.output.weight.normal_(std=0.05)

    model.factorized_landmark_budgeted = False
    unbudgeted = model.forward_navigation(**inputs)
    base = torch.softmax(unbudgeted['base_global_logits'][:, 3:], -1)
    routed = unbudgeted['global_logits'][:, 3:].exp()
    routed = routed / routed.sum(-1, keepdim=True)
    costs = (routed > base).to(inputs['gmap_pair_dists'].dtype)
    inputs['gmap_pair_dists'][:, 2, 3:] = costs
    model.factorized_landmark_budgeted = True

    outputs = model.forward_navigation(**inputs)

    assert outputs['budget_activation_rate'].item() == 1.0
    assert outputs['projected_route_cost'] <= outputs['base_route_cost'] + 1e-6
    assert outputs['routed_route_cost'] > outputs['base_route_cost']
    assert torch.allclose(
        outputs['global_logits'].exp()[:, 0],
        torch.softmax(outputs['base_global_logits'], -1)[:, 0],
        atol=1e-7,
    )
    outputs['global_logits'][:, 3].sum().backward()
    assert model.factorized_landmark.output.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith('factorized_landmark.')
    )


def test_budget_mode_requires_factorized_landmark_module(model):
    config = SimpleNamespace(**vars(model.config))
    config.factorized_landmark_size = 0
    with pytest.raises(ValueError, match='requires factorized landmark routing'):
        type(model)(config)
