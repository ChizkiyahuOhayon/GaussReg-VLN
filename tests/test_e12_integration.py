"""Run actual graph attention and navigation code with a small model config.

Only the unavailable HuggingFace constructor is shimmed. Server smoke exercises
its real pretrained loader, 768-D production configuration and strict E0 weights.
"""

import importlib.util
from collections import defaultdict
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_and_smoke(monkeypatch):
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
    for name, module in [('vlnce_baselines', package),
                         ('vlnce_baselines.common', common),
                         ('transformers', transformers)]:
        monkeypatch.setitem(sys.modules, name, module)
    # Explicit loads keep imports isolated from the Habitat package initializer.
    load('vlnce_baselines.common.transformer', 'vlnce_baselines/common/transformer.py')
    load('vlnce_baselines.common.ops', 'vlnce_baselines/common/ops.py')
    load('vlnce_baselines.geo_token', 'vlnce_baselines/geo_token.py')
    load('vlnce_baselines.successor', 'vlnce_baselines/successor.py')
    module = load('e12_real_forward', 'vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py')
    config = SimpleNamespace(
        hidden_size=16, vocab_size=32, max_position_embeddings=32,
        type_vocab_size=2, max_txt_task_embeddings=4, layer_norm_eps=1e-12,
        hidden_dropout_prob=0., num_attention_heads=4, output_attentions=True,
        attention_probs_dropout_prob=0., intermediate_size=32, hidden_act='gelu',
        num_l_layers=1, update_lang_bert=False, image_feat_size=8,
        use_depth_embedding=True, depth_feat_size=4, angle_feat_size=4,
        num_pano_layers=1, max_action_steps=100, max_gmap_task_embeddings=3,
        num_x_layers=2, use_lang2visn_attn=True, graph_sprels=True,
        pred_head_dropout_prob=0., fix_lang_embedding=True, fix_pano_embedding=True,
        successor_hidden_size=8,
    )
    torch.manual_seed(11)
    model = module.GlocalTextPathNavCMT(config)
    smoke = load('e12_smoke_for_test', 'tools/smoke_e12_model.py')
    return model, smoke


def test_actual_forward_has_only_real_actions_and_trains_only_successor(model_and_smoke):
    model, smoke = model_and_smoke
    assert smoke.exercise_model(model) > 0


def test_actual_forward_roundtrips_full_state_and_rejects_partial_state(model_and_smoke):
    model, _ = model_and_smoke
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    del state['successor.output.weight']
    with pytest.raises(RuntimeError, match='successor.output.weight'):
        model.load_state_dict(state, strict=True)


def test_replay_update_changes_decoder_and_preserves_all_e0_weights(model_and_smoke):
    model, _ = model_and_smoke
    module = sys.modules['vlnce_baselines.successor']
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('successor.' in name)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.vln_bert = model

        def forward(self, mode=None, gmap_vp_ids=None, **kwargs):
            return model.forward_navigation(gmap_vpids=gmap_vp_ids, **kwargs)

    nav = dict(
        mode='navigation', gmap_vp_ids=[[None, '0', 'g0', 'g1']],
        gmap_step_ids=torch.tensor([[0, 1, 0, 0]]),
        gmap_img_fts=torch.randn(1, 4, 16), gmap_pos_fts=torch.randn(1, 4, 7),
        gmap_masks=torch.ones(1, 4, dtype=torch.bool),
        gmap_visited_masks=torch.tensor([[False, True, False, False]]),
        gmap_pair_dists=torch.zeros(1, 4, 4), gmap_task_embeddings=torch.ones(1, 4, dtype=torch.long),
    )
    following = dict(nav)
    following.update(gmap_step_ids=torch.tensor([[0, 1, 2, 0]]),
                     gmap_visited_masks=torch.tensor([[False, True, True, False]]),
                     gmap_img_fts=torch.randn(1, 4, 16))
    steps = [dict(input=nav, indices=[0], action=torch.tensor([3]),
                  executed_move=torch.tensor([True]),
                  current_position=torch.zeros(1, 3), successor_position=torch.ones(1, 3)),
             dict(input=following, indices=[0], action=torch.tensor([0]),
                  executed_move=torch.tensor([False]), current_position=torch.ones(1, 3))]
    trainer = SimpleNamespace(
        policy=SimpleNamespace(net=PolicyNet()), device='cpu', max_grad_norm=2.,
        config=SimpleNamespace(GRPO=SimpleNamespace(loc_noise=0.5)),
        data_buffer=[dict(data_buffer=steps, initial_txt_embeds=torch.randn(1, 3, 16),
                          initial_txt_masks=torch.ones(1, 3, dtype=torch.bool))],
        logs=defaultdict(list), set_policy_mode=lambda mode: None,
        optimizer=torch.optim.AdamW(model.successor.parameters(), lr=1e-3),
    )
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.)
    module.update_successor(trainer)
    assert model.successor.updates == 1
    assert not trainer.data_buffer
    assert trainer.logs['successor_pairs'] == [1]
    assert trainer.logs['grad_norm'][0] > 0
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith('successor.') and k != 'successor.updates')
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith('successor.'))
