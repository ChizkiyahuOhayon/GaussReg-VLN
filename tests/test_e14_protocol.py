import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'e14_protocol', ROOT / 'tools/e14_protocol.py'
)
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def _row(**changes):
    row = dict(distance_to_goal=3.8, oracle_success=1., success=1.,
               spl=0.6, ndtw=0.7, sdtw=0.6)
    row.update(changes)
    return row


def test_gate_requires_all_episodes_and_all_six_metrics():
    rows = {'1': _row(), '2': _row()}
    assert PROTOCOL.evaluate_results(rows, {'1', '2'})['decision'] == 'go'
    rows['2']['spl'] = 0.5
    assert PROTOCOL.evaluate_results(rows, {'1', '2'})['decision'] == 'no-go'
    with pytest.raises(ValueError, match='episode'):
        PROTOCOL.evaluate_results(rows, {'1', '3'})


def _config():
    grpo = SimpleNamespace(
        landmark_transport_only=True,
        gauss_only=False, candidate_scorer_only=False,
        gaussian_bev_only=False, anchor_repair_only=False,
        hindsight_stop_only=False, terminal_commit_only=False,
        success_set_commit=False, setwise_group_policy=False,
        frontier_advantage=False, geo_token_only=False,
        successor_only=False, instruction_coverage_only=False,
        sample_num=8, update_epochs=1, lr=1e-4, grpo_beta=0.04,
        back_algo='control', is_requeue=False, enable_amp=False,
        enable_all_dropouts=False, dropout_in_sampling=False,
        waypoint_aug=False,
    )
    return SimpleNamespace(
        GRPO=grpo,
        MODEL=SimpleNamespace(
            gauss_feat_size=0, candidate_scorer_hidden_size=0,
            gaussian_bev_hidden_size=0, anchor_repair_hidden_size=0,
            hindsight_stop_hidden_size=0, terminal_commit_hidden_size=0,
            geo_token_hidden_size=0, successor_hidden_size=0,
            instruction_coverage_hidden_size=0, landmark_transport_size=128,
        ),
        TASK_CONFIG=SimpleNamespace(
            DATASET=SimpleNamespace(SPLIT='train', SUFFIX='_10')
        ),
    )


@pytest.mark.parametrize('damage', [None, 'frozen', 'partial', 'unchanged',
                                     'no_updates', 'config'])
def test_checkpoint_gate_enforces_e14_contract(monkeypatch, tmp_path, damage):
    spec = importlib.util.spec_from_file_location(
        'landmark_transport_fixture',
        ROOT / 'vlnce_baselines/landmark_transport.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transport = module.LandmarkTransport(768, 128).state_dict()
    transport['output.bias'].add_(0.1)
    state = {'net.vln_bert.landmark_transport.' + key: value
             for key, value in transport.items()}
    state['net.frozen'] = torch.ones(2)
    checkpoint = dict(state_dict=state, iteration=20,
                      e14_optimizer_updates=20, config=_config())
    if damage == 'frozen':
        state['net.frozen'].zero_()
    elif damage == 'partial':
        del state['net.vln_bert.landmark_transport.view_projection.weight']
    elif damage == 'unchanged':
        transport['output.bias'].zero_()
    elif damage == 'no_updates':
        checkpoint['e14_optimizer_updates'] = 19
    elif damage == 'config':
        checkpoint['config'].GRPO.grpo_beta = 0.0

    path = tmp_path / 'checkpoint.pth'
    path.write_bytes(b'fixture; torch loading is stubbed')
    monkeypatch.setattr(
        torch, 'load',
        lambda source, **kwargs: checkpoint if source == path else {
            'state_dict': {'net.frozen': torch.ones(2)}
        },
    )
    args = SimpleNamespace(checkpoint=path, baseline='baseline.pth',
                           iteration=20)
    if damage is None:
        PROTOCOL.check_checkpoint(args)
    else:
        with pytest.raises(RuntimeError):
            PROTOCOL.check_checkpoint(args)
