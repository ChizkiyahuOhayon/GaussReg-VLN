"""Exercise E13 evidence extraction and navigation with the real model code."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


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
    load(
        'vlnce_baselines.common.transformer',
        'vlnce_baselines/common/transformer.py',
    )
    load('vlnce_baselines.common.ops', 'vlnce_baselines/common/ops.py')
    load('vlnce_baselines.geo_token', 'vlnce_baselines/geo_token.py')
    load(
        'vlnce_baselines.instruction_coverage',
        'vlnce_baselines/instruction_coverage.py',
    )
    module = load(
        'e13_real_forward',
        'vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py',
    )
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
        fix_pano_embedding=True, instruction_coverage_hidden_size=32,
        successor_hidden_size=0,
    )
    torch.manual_seed(13)
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
        gmap_instruction_evidence=torch.rand(2, 5, 4),
    )


def test_prepool_evidence_is_detached_normalized_and_masks_views(model):
    views = torch.randn(2, 5, 16, requires_grad=True)
    view_masks = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, False],
    ])
    inputs = navigation_inputs()
    evidence, slot_masks = model.forward_instruction_evidence(
        views, view_masks, inputs['txt_embeds'], inputs['txt_masks']
    )

    assert evidence.shape == (2, 5, 4)
    assert not evidence.requires_grad
    assert torch.allclose(
        evidence.sum(dim=-1)[view_masks],
        torch.ones_like(evidence.sum(dim=-1)[view_masks]),
    )
    assert torch.count_nonzero(evidence[view_masks.logical_not()]) == 0
    assert slot_masks.tolist() == [[True] * 4, [True] * 4]


def test_zero_residual_is_exact_e0_and_only_head_receives_gradient(model):
    inputs = navigation_inputs()
    enabled = model.forward_navigation(**inputs)
    coverage_module = model.instruction_coverage
    model.instruction_coverage = None
    disabled = model.forward_navigation(**{
        key: value for key, value in inputs.items()
        if key != 'gmap_instruction_evidence'
    })
    model.instruction_coverage = coverage_module

    assert torch.equal(enabled['global_logits'], disabled['global_logits'])
    assert torch.equal(
        torch.isfinite(enabled['global_logits']),
        torch.isfinite(disabled['global_logits']),
    )
    assert torch.count_nonzero(enabled['instruction_coverage_residual']) == 0

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.instruction_coverage.parameters():
        parameter.requires_grad_(True)
    model.forward_navigation(**inputs)['global_logits'][:, 3].sum().backward()
    assert model.instruction_coverage.output.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith('instruction_coverage.')
    )


def test_full_e13_state_roundtrips_and_partial_state_is_rejected(model):
    state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    del state['instruction_coverage.output.weight']
    with pytest.raises(RuntimeError, match='instruction_coverage.output.weight'):
        model.load_state_dict(state, strict=True)


def test_model_rejects_combining_e13_with_gaussian_features(model):
    config = SimpleNamespace(**vars(model.config))
    config.gauss_feat_size = 5
    with pytest.raises(ValueError, match='E2-E12 modules off'):
        type(model)(config)
