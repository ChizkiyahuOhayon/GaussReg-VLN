import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e0_equivalence_uses_probability_precision_not_log_roundoff():
    factorized = load(
        'factorized_landmark_smoke_fixture',
        'vlnce_baselines/factorized_landmark.py',
    )
    smoke = load('e16_smoke_for_test', 'tools/smoke_e16_model.py')
    torch.manual_seed(16)
    base = torch.randn(8, 512)
    masks = torch.rand(8, 512) > 0.6
    masks[:, 0] = False
    masks[:, 1] = True
    routed = base.masked_fill(masks.logical_not(), -float('inf'))
    actual = factorized.factorized_action_log_probs(base, routed, masks)
    expected = torch.log_softmax(base, dim=-1)

    finite = torch.isfinite(expected)
    assert (actual[finite] - expected[finite]).abs().max() > 1e-7
    mismatch, tolerance = smoke.probability_mismatch(actual, base)
    assert mismatch <= tolerance
