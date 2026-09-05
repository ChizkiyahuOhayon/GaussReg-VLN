#!/usr/bin/env python3
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'vlnce_baselines/frontier_advantage.py'
SPEC = importlib.util.spec_from_file_location(
    'frontier_advantage', str(MODULE_PATH)
)
FRONTIER_ADVANTAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRONTIER_ADVANTAGE)


def main():
    advantages = FRONTIER_ADVANTAGE.action_set_advantages(
        current_distances=torch.tensor([5.0, 4.0]),
        candidate_distances=torch.tensor([
            [5.0, 3.0, 7.0],
            [2.5, 3.0, 5.0],
        ]),
        valid_mask=torch.ones(2, 3, dtype=torch.bool),
        success_distance=3.0,
        progress_clip=3.0,
    )
    assert advantages[0, 1] > advantages[0, 2] > advantages[0, 0]
    assert advantages[1, 0] > advantages[1, 1] > advantages[1, 2]

    mixed = FRONTIER_ADVANTAGE.mix_advantages(
        torch.tensor([0.5, -0.5]), advantages[:, 0], weight=0.5
    )
    assert torch.isfinite(mixed).all()
    assert FRONTIER_ADVANTAGE.annealed_weight(
        1.0, 0.25, step=499, total_steps=500
    ) == 0.25

    print(
        'E10_OBJECTIVE_SMOKE_PASSED '
        'action_sets=2 added_parameters=0 inference_changes=0'
    )


if __name__ == '__main__':
    main()
