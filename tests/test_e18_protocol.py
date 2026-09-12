"""E18 evidence integrity and strict numerical boundary tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    spec = importlib.util.spec_from_file_location('e18_protocol_test', ROOT / 'tools/e18_protocol.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(spl=.57):
    return {'1': dict(distance_to_goal=3.8, oracle_success=1, success=1, spl=spl,
                      ndtw=.7, sdtw=.6)}


def test_gate_requires_spl_above_5685_and_six_e15_improvements():
    p = protocol()
    assert p.evaluate_results(rows(), {'1'})['decision'] == 'go'
    assert p.evaluate_results(rows(.5685), {'1'})['decision'] == 'no-go'
    data = rows()
    data['1']['distance_to_goal'] = p.E15_REFERENCE['NE']
    assert p.evaluate_results(data, {'1'})['decision'] == 'no-go'
    with pytest.raises(ValueError):
        p.evaluate_results(rows(), {'2'})
    data = rows()
    data['1']['success'] = .5
    with pytest.raises(ValueError, match='binary'):
        p.evaluate_results(data, {'1'})


def test_checkpoint_requires_exact_e0_frozen_and_changed_complete_clone():
    p = protocol()
    frozen = {'net.vln_bert.global_encoder.encoder.x_layers.2.weight': torch.ones(2),
              'net.vln_bert.graph_query_text.weight': torch.ones(2),
              'net.vln_bert.graph_attentioned_txt_embeds_transform.weight': torch.ones(2),
              'net.vln_bert.global_sap_head.weight': torch.ones(2),
              'net.other': torch.ones(1)}
    state = {k: v.clone() for k, v in frozen.items()}
    mapping = p.clone_key_mapping(frozen)
    state.update({k: frozen[v].clone() for k, v in mapping.items()})
    with pytest.raises(RuntimeError, match='unchanged'):
        p.validate_state(state, frozen)
    state[next(iter(mapping))] += .1
    assert p.validate_state(state, frozen)['trainable_parameters'] == 8
    state['net.other'] += 1
    with pytest.raises(RuntimeError, match='frozen E0'):
        p.validate_state(state, frozen)
    state['net.other'] -= 1
    state[next(iter(mapping))][0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        p.validate_state(state, frozen)
    state.pop(next(iter(mapping)))
    with pytest.raises(RuntimeError, match='complete'):
        p.validate_state(state, frozen)
