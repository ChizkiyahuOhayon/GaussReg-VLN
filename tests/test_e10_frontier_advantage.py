import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] /
    'vlnce_baselines/frontier_advantage.py'
)
SPEC = importlib.util.spec_from_file_location(
    'frontier_advantage_for_test', str(MODULE_PATH)
)
FRONTIER_ADVANTAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRONTIER_ADVANTAGE)
action_set_advantages = FRONTIER_ADVANTAGE.action_set_advantages
annealed_weight = FRONTIER_ADVANTAGE.annealed_weight
mix_advantages = FRONTIER_ADVANTAGE.mix_advantages


def test_action_set_advantages_rank_progress_and_mask_invalid_actions():
    advantages = action_set_advantages(
        current_distances=torch.tensor([5.0]),
        candidate_distances=torch.tensor([[5.0, 3.0, 7.0, float('inf')]]),
        valid_mask=torch.tensor([[True, True, True, False]]),
        success_distance=3.0,
        progress_clip=3.0,
    )

    assert advantages[0, 1] > advantages[0, 2] > advantages[0, 0]
    assert advantages[0, 3] == 0.0
    assert torch.allclose(advantages[0, :3].mean(), torch.tensor(0.0))
    assert torch.allclose(
        advantages[0, :3].pow(2).mean(), torch.tensor(1.0), atol=1e-6
    )


def test_stop_utility_uses_success_boundary():
    successful_stop = action_set_advantages(
        torch.tensor([4.0]),
        torch.tensor([[2.5, 3.0]]),
        torch.tensor([[True, True]]),
        success_distance=3.0,
        progress_clip=3.0,
    )
    failed_stop = action_set_advantages(
        torch.tensor([4.0]),
        torch.tensor([[3.5, 3.0]]),
        torch.tensor([[True, True]]),
        success_distance=3.0,
        progress_clip=3.0,
    )

    assert successful_stop[0, 0] > successful_stop[0, 1]
    assert failed_stop[0, 0] < failed_stop[0, 1]


def test_degenerate_action_sets_have_zero_local_advantage():
    single = action_set_advantages(
        torch.tensor([5.0]),
        torch.tensor([[5.0, 4.0]]),
        torch.tensor([[True, False]]),
        success_distance=3.0,
        progress_clip=3.0,
    )
    tied = action_set_advantages(
        torch.tensor([5.0]),
        torch.tensor([[4.0, 4.0]]),
        torch.tensor([[False, True]]),
        success_distance=3.0,
        progress_clip=3.0,
    )

    assert torch.count_nonzero(single) == 0
    assert torch.count_nonzero(tied) == 0


def test_mix_advantages_and_linear_schedule():
    trajectory = torch.tensor([-1.0, 0.5])
    local = torch.tensor([0.25, -0.5])

    assert torch.equal(
        mix_advantages(trajectory, local, weight=0.0), trajectory
    )
    assert torch.equal(
        mix_advantages(trajectory, local, weight=2.0),
        torch.tensor([-0.5, -0.5]),
    )
    assert annealed_weight(1.0, 0.25, step=0, total_steps=500) == 1.0
    assert annealed_weight(1.0, 0.25, step=499, total_steps=500) == 0.25
    assert annealed_weight(1.0, 0.25, step=999, total_steps=500) == 0.25
