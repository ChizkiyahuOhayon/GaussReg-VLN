import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO_ROOT / 'vlnce_baselines/setwise_policy.py'
    spec = importlib.util.spec_from_file_location('setwise_policy', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_successful_rollouts_always_rank_above_failures():
    module = _load_module()
    outcomes = [
        [(False, 0.0, 0.1)],
        [(True, 0.2, 2.9)],
        [(False, 0.0, 5.0)],
    ]

    advantages, active_groups = module.setwise_group_advantages(outcomes)

    assert active_groups == [True]
    assert advantages[1][0] > advantages[0][0] > advantages[2][0]


def test_successes_rank_by_spl_and_failures_by_nearest_distance():
    module = _load_module()
    outcomes = [
        [(True, 0.6, 1.0), (False, 0.0, 4.0)],
        [(True, 0.9, 2.5), (False, 0.0, 1.5)],
        [(True, 0.7, 0.5), (False, 0.0, 3.0)],
    ]

    advantages, active_groups = module.setwise_group_advantages(outcomes)

    assert active_groups == [True, True]
    assert advantages[1][0] > advantages[2][0] > advantages[0][0]
    assert advantages[1][1] > advantages[2][1] > advantages[0][1]


def test_ties_share_rank_and_constant_groups_are_inactive():
    module = _load_module()
    outcomes = [
        [(True, 0.8, 1.0), (False, 0.0, 2.0)],
        [(True, 0.8, 2.5), (False, 0.0, 2.0)],
        [(False, 0.0, 0.1), (False, 0.0, 2.0)],
    ]

    advantages, active_groups = module.setwise_group_advantages(outcomes)

    assert advantages[0][0] == pytest.approx(advantages[1][0])
    assert advantages[0][0] > advantages[2][0]
    assert active_groups == [True, False]
    assert [row[1] for row in advantages] == [0.0, 0.0, 0.0]


def test_recoverable_outcome_uses_the_shortest_successful_prefix():
    module = _load_module()
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ])
    distances = np.array([6.0, 3.5, 2.5, 1.0])

    outcome = module.recoverable_outcome(
        positions, distances, success_distance=3.0
    )

    assert outcome[0] is True
    assert outcome[1] == pytest.approx(1.0)
    assert outcome[2] == pytest.approx(1.0)


def test_recoverable_outcome_reports_failure_by_closest_distance():
    module = _load_module()
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    distances = np.array([6.0, 3.1])

    outcome = module.recoverable_outcome(
        positions, distances, success_distance=3.0
    )

    assert outcome == (False, 0.0, 3.1)
