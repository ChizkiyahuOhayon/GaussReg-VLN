#!/usr/bin/env python3
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'vlnce_baselines/setwise_policy.py'
SPEC = importlib.util.spec_from_file_location('setwise_policy', str(MODULE_PATH))
SETWISE_POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETWISE_POLICY)
recoverable_outcome = SETWISE_POLICY.recoverable_outcome
setwise_group_advantages = SETWISE_POLICY.setwise_group_advantages


def main():
    outcomes = [
        [(False, 0.0, 4.0)],
        [(True, 0.55, 2.8)],
        [(False, 0.0, 2.0)],
        [(True, 0.80, 1.5)],
        [(True, 0.80, 2.0)],
        [(False, 0.0, 3.0)],
        [(True, 0.65, 2.5)],
        [(False, 0.0, 5.0)],
    ]
    advantages, active_groups = setwise_group_advantages(outcomes)
    values = [sample[0] for sample in advantages]

    assert active_groups == [True]
    assert values[3] == values[4]
    assert min(values[index] for index in (1, 3, 4, 6)) > max(
        values[index] for index in (0, 2, 5, 7)
    )

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])
    outcome = recoverable_outcome(
        positions, np.array([6.0, 3.5, 2.5]), success_distance=3.0
    )
    assert outcome == (True, 1.0, 2.5)

    print(
        'E9_OBJECTIVE_SMOKE_PASSED '
        'samples=8 active_groups=1 added_parameters=0'
    )


if __name__ == '__main__':
    main()
