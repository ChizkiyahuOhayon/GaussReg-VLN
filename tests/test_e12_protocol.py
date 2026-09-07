import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('e12_protocol', ROOT / 'tools/e12_protocol.py')
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def _row(**changes):
    row = dict(distance_to_goal=3.8, oracle_success=1., success=1.,
               spl=0.6, ndtw=0.7, sdtw=0.6)
    row.update(changes)
    return row


def test_gate_requires_matching_episode_ids_and_all_six_metrics():
    rows = {'1': _row(), '2': _row()}
    report = PROTOCOL.evaluate_results(rows, {'1', '2'})
    assert report['decision'] == 'go'
    rows['2']['spl'] = 0.5
    assert PROTOCOL.evaluate_results(rows, {'1', '2'})['decision'] == 'no-go'
    with pytest.raises(ValueError, match='episode'):
        PROTOCOL.evaluate_results(rows, {'1', '3'})


def test_nan_or_fraction_scale_mistakes_do_not_become_results():
    for value in (float('nan'), float('inf'), 65.8):
        with pytest.raises(ValueError):
            PROTOCOL.evaluate_results({'1': _row(ndtw=value)}, {'1'})


def test_thresholds_are_strict_and_path_fidelity_is_required():
    assert PROTOCOL.evaluate_results({'1': _row(distance_to_goal=3.93)}, {'1'})['decision'] == 'no-go'
    assert PROTOCOL.evaluate_results({'1': _row(ndtw=0.65)}, {'1'})['decision'] == 'no-go'
    assert PROTOCOL.evaluate_results({'1': _row(sdtw=0.53)}, {'1'})['decision'] == 'no-go'


@pytest.mark.parametrize('damage', [None, 'frozen', 'partial', 'nonfinite', 'unchanged', 'no_updates'])
def test_checkpoint_gate_rejects_invalid_training_evidence(monkeypatch, tmp_path, damage):
    spec = importlib.util.spec_from_file_location('e12_decoder_check_test', ROOT / 'vlnce_baselines/successor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = module.SuccessorDecoder(768).state_dict()
    initial = module.decoder_digest(decoder)
    decoder['output.bias'].add_(0.1)
    decoder['updates'].fill_(20)
    state = {'net.vln_bert.successor.' + key: value for key, value in decoder.items()}
    state['net.frozen'] = torch.ones(2)
    checkpoint = dict(
        state_dict=state, iteration=20, e12_initial_decoder_sha256=initial,
        config=SimpleNamespace(
            GRPO=SimpleNamespace(successor_only=True, back_algo='control', is_requeue=False),
            MODEL=SimpleNamespace(successor_hidden_size=256),
            TASK_CONFIG=SimpleNamespace(DATASET=SimpleNamespace(SPLIT='train', SUFFIX='_10')),
        ),
    )
    if damage == 'frozen':
        state['net.frozen'].zero_()
    elif damage == 'partial':
        del state['net.vln_bert.successor.output.weight']
    elif damage == 'nonfinite':
        decoder['output.bias'][0] = float('nan')
    elif damage == 'unchanged':
        checkpoint['e12_initial_decoder_sha256'] = module.decoder_digest(decoder)
    elif damage == 'no_updates':
        decoder['updates'].zero_()
    path = tmp_path / 'checkpoint.pth'
    path.write_bytes(b'fixture bytes for hash; loading is stubbed')
    monkeypatch.setattr(torch, 'load', lambda source, **kwargs:
                        checkpoint if source == path else {'state_dict': {'net.frozen': torch.ones(2)}})
    args = SimpleNamespace(checkpoint=path, baseline='baseline.pth', iteration=20)
    if damage is None:
        PROTOCOL.check_checkpoint(args)
    else:
        with pytest.raises(RuntimeError):
            PROTOCOL.check_checkpoint(args)
