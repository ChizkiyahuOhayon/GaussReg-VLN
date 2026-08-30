import torch


def counterfactual_stop_returns(
    goal_distances,
    rollback_distances,
    path_length,
    shortest_path_length,
    success_distance,
    distance_scale,
):
    """Compute the terminal R2R reward for stopping at visited nodes."""
    final_path_length = path_length.unsqueeze(1) + rollback_distances
    success = (goal_distances <= success_distance).to(goal_distances.dtype)
    shortest_path_length = shortest_path_length.unsqueeze(1)
    spl = success * shortest_path_length / torch.maximum(
        shortest_path_length, final_path_length
    ).clamp_min(1e-6)
    distance_reward = 1.0 - goal_distances / distance_scale
    return spl + success + distance_reward


def hindsight_stop_targets(stop_returns, continue_returns, valid_mask):
    """Select CONTINUE (index zero) or the best valid visited node."""
    returns = stop_returns.clone()
    returns[:, 0] = continue_returns
    returns.masked_fill_(valid_mask.logical_not(), -float('inf'))
    return returns.argmax(dim=1)
