import torch


def action_set_advantages(
    current_distances,
    candidate_distances,
    valid_mask,
    success_distance,
    progress_clip,
    eps=1e-8,
):
    """Standardize one-step geodesic utility within each action set.

    Action zero is ETP's STOP action. Its utility is the positive or negative
    clipping boundary according to whether the rollback destination succeeds.
    Other actions use clipped geodesic progress toward the goal.
    """
    current_distances = current_distances.to(candidate_distances)
    valid_mask = valid_mask.bool()
    finite_mask = (
        torch.isfinite(current_distances).unsqueeze(1) &
        torch.isfinite(candidate_distances)
    )
    valid_mask = valid_mask & finite_mask

    utilities = torch.clamp(
        current_distances.unsqueeze(1) - candidate_distances,
        min=-progress_clip,
        max=progress_clip,
    )
    stop_utility = torch.where(
        candidate_distances[:, 0] <= success_distance,
        utilities.new_full((utilities.size(0),), progress_clip),
        utilities.new_full((utilities.size(0),), -progress_clip),
    )
    utilities = utilities.clone()
    utilities[:, 0] = stop_utility

    weights = valid_mask.to(utilities.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    safe_utilities = torch.where(
        valid_mask, utilities, torch.zeros_like(utilities)
    )
    means = safe_utilities.sum(dim=1, keepdim=True) / counts.clamp_min(1)
    centered = torch.where(
        valid_mask, utilities - means, torch.zeros_like(utilities)
    )
    variances = (
        centered.square() * weights
    ).sum(dim=1, keepdim=True) / counts.clamp_min(1)
    active = (counts > 1) & (variances > eps)
    advantages = centered / torch.sqrt(variances + eps)
    return torch.where(valid_mask & active, advantages, torch.zeros_like(advantages))


def mix_advantages(trajectory_advantages, local_advantages, weight):
    return trajectory_advantages + weight * local_advantages


def annealed_weight(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    progress = min(max(step, 0), total_steps - 1) / float(total_steps - 1)
    return float(start + progress * (end - start))
