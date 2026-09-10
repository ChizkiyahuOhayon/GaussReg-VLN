"""Minimum-KL graph-cost projection for factorized landmark routing."""

import torch
from torch.nn import functional as F


_SOLVER_STEPS = 32
_COST_TOLERANCE = 1e-6


def current_frontier_costs(step_ids, pair_dists, visited_masks):
    """Return graph distances from the most recently visited node."""
    if (step_ids.ndim != 2 or visited_masks.shape != step_ids.shape or
            pair_dists.shape != step_ids.shape + (step_ids.size(1),)):
        raise ValueError('Graph state tensors are misaligned')
    current = step_ids.masked_fill(visited_masks.logical_not(), -1).argmax(1)
    rows = torch.arange(step_ids.size(0), device=step_ids.device)
    return pair_dists[rows, current]


def _expected_cost(logits, masks, costs):
    log_probs = F.log_softmax(
        logits.masked_fill(masks.logical_not(), -float('inf')), dim=-1
    )
    probabilities = torch.where(
        masks, log_probs.exp(), torch.zeros_like(log_probs)
    )
    return (probabilities * costs).sum(-1)


def project_frontier_logits(base_logits, routed_logits, frontier_masks,
                            frontier_costs):
    """Project routed logits into the frozen E0 expected graph-cost budget."""
    if (base_logits.ndim != 2 or base_logits.size(1) == 0 or
            routed_logits.shape != base_logits.shape or
            frontier_masks.shape != base_logits.shape or
            frontier_costs.shape != base_logits.shape):
        raise ValueError('Budgeted routing tensors are misaligned')
    if frontier_masks.dtype != torch.bool:
        raise ValueError('Frontier masks must be boolean')
    valid_costs = frontier_costs.masked_select(frontier_masks)
    if (not torch.isfinite(valid_costs).all() or
            (valid_costs < 0).any()):
        raise ValueError('Valid frontier costs must be finite and nonnegative')

    projected = routed_logits
    projected_costs = routed_logits.new_zeros(routed_logits.size(0))
    rows = frontier_masks.any(-1)
    if not rows.any():
        return projected, {
            'activation_rate': routed_logits.new_zeros(()),
            'dual_lambda': routed_logits.new_zeros(()),
            'base_cost': routed_logits.new_zeros(()),
            'routed_cost': routed_logits.new_zeros(()),
            'projected_cost': routed_logits.new_zeros(()),
        }

    masks = frontier_masks[rows]
    costs = frontier_costs[rows].to(routed_logits.dtype)
    base_rows = base_logits[rows]
    routed_rows = routed_logits[rows]
    with torch.no_grad():
        budgets = _expected_cost(base_rows, masks, costs)
        unconstrained = _expected_cost(routed_rows, masks, costs)
        cost_range = (
            costs.masked_fill(masks.logical_not(), -float('inf')).max(-1).values -
            costs.masked_fill(masks.logical_not(), float('inf')).min(-1).values
        )
        active = (
            (unconstrained > budgets + _COST_TOLERANCE) &
            (cost_range > _COST_TOLERANCE)
        )
        row_count = rows.sum().to(routed_logits.dtype)
        if not active.any():
            return projected, {
                'activation_rate': routed_logits.new_zeros(()),
                'dual_lambda': routed_logits.new_zeros(()),
                'base_cost': budgets.sum() / row_count,
                'routed_cost': unconstrained.sum() / row_count,
                'projected_cost': unconstrained.sum() / row_count,
            }
        low = torch.zeros_like(budgets)
        high = torch.ones_like(budgets)
        for _ in range(_SOLVER_STEPS):
            high_cost = _expected_cost(
                routed_rows - high.unsqueeze(-1) * costs, masks, costs
            )
            high = torch.where(active & (high_cost > budgets), high * 2, high)
        for _ in range(_SOLVER_STEPS):
            middle = (low + high) * 0.5
            middle_cost = _expected_cost(
                routed_rows - middle.unsqueeze(-1) * costs, masks, costs
            )
            too_expensive = active & (middle_cost > budgets)
            low = torch.where(too_expensive, middle, low)
            high = torch.where(active & too_expensive.logical_not(), middle, high)
        solved = torch.where(active, high, torch.zeros_like(high))

    tilted = routed_rows - solved.unsqueeze(-1) * costs
    projected_rows = torch.where(active.unsqueeze(-1), tilted, routed_rows)
    projected = projected.clone()
    projected[rows] = projected_rows
    projected_costs[rows] = _expected_cost(
        projected_rows.detach(), masks, costs
    )
    return projected, {
        'activation_rate': active.to(routed_logits.dtype).mean(),
        'dual_lambda': solved.sum() / row_count,
        'base_cost': budgets.sum() / row_count,
        'routed_cost': unconstrained.sum() / row_count,
        'projected_cost': projected_costs.sum() / row_count,
    }
