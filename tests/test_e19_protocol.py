import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'e19_protocol', ROOT / 'tools/e19_protocol.py'
)
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def row(**changes):
    value = dict(distance_to_goal=3.8, oracle_success=1., success=1.,
                 spl=0.6, ndtw=0.7, sdtw=0.6)
    value.update(changes)
    return value


def test_gate_requires_all_episodes_and_all_six_metrics():
    rows = {'1': row(), '2': row()}
    assert PROTOCOL.evaluate_results(rows, {'1', '2'})['decision'] == 'go'
    rows['2']['spl'] = 0.5
    assert PROTOCOL.evaluate_results(rows, {'1', '2'})['decision'] == 'no-go'
    with pytest.raises(ValueError, match='episode'):
        PROTOCOL.evaluate_results(rows, {'1', '3'})


def config():
    excluded = {
        name: False for name in [
            'gauss_only', 'candidate_scorer_only', 'gaussian_bev_only',
            'anchor_repair_only', 'hindsight_stop_only',
            'terminal_commit_only', 'success_set_commit',
            'setwise_group_policy', 'frontier_advantage', 'geo_token_only',
            'successor_only', 'instruction_coverage_only',
            'landmark_transport_only', 'factorized_landmark_only',
            'budgeted_factorized_landmark_only',
            'monotonic_factorized_landmark_only', 'route_attention_only',
        ]
    }
    grpo = SimpleNamespace(
        **excluded, factorized_geometry_only=True, sample_num=8,
        update_epochs=1, lr=1e-4, grpo_beta=0.04, back_algo='control',
        is_requeue=False, enable_amp=False, enable_all_dropouts=False,
        dropout_in_sampling=False, waypoint_aug=False,
    )
    model = SimpleNamespace(
        factorized_landmark_size=128, transient_geometry_size=128,
        factorized_landmark_budgeted=False,
        factorized_landmark_monotonic=False, route_attention=False,
    )
    return SimpleNamespace(
        GRPO=grpo, MODEL=model,
        TASK_CONFIG=SimpleNamespace(
            DATASET=SimpleNamespace(SPLIT='train', SUFFIX='_10')
        ),
    )


@pytest.mark.parametrize('damage', [None, 'frozen', 'partial', 'unchanged',
                                     'no_updates', 'config'])
def test_checkpoint_gate_enforces_e19_contract(monkeypatch, tmp_path, damage):
    landmark = PROTOCOL.load_module(
        'e19_landmark_fixture', 'vlnce_baselines/landmark_transport.py'
    ).LandmarkTransport(768, 128).state_dict()
    geometry = PROTOCOL.load_module(
        'e19_geometry_fixture', 'vlnce_baselines/transient_local_geometry.py'
    ).TransientLocalGeometry(768, 128).state_dict()
    landmark['output.bias'].add_(0.1)
    geometry['output.bias'].add_(0.1)
    state = {
        'net.vln_bert.factorized_landmark.' + key: value
        for key, value in landmark.items()
    }
    state.update({
        'net.vln_bert.transient_geometry.' + key: value
        for key, value in geometry.items()
    })
    state['net.frozen'] = torch.ones(2)
    checkpoint = dict(state_dict=state, iteration=20,
                      e19_optimizer_updates=20, config=config())
    if damage == 'frozen':
        state['net.frozen'].zero_()
    elif damage == 'partial':
        del state['net.vln_bert.transient_geometry.point_projection.weight']
    elif damage == 'unchanged':
        geometry['output.bias'].zero_()
    elif damage == 'no_updates':
        checkpoint['e19_optimizer_updates'] = 19
    elif damage == 'config':
        checkpoint['config'].MODEL.transient_geometry_size = 64

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
