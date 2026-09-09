"""Landmark routing that preserves the frozen policy's stop factor."""

import torch
from torch.nn import functional as F


def frontier_action_mask(action_masks, visited_masks):
    """Return valid unvisited non-STOP actions."""
    if action_masks.shape != visited_masks.shape:
        raise ValueError('Action and visited masks are misaligned')
    if action_masks.ndim != 2 or action_masks.size(1) == 0:
        raise ValueError('Action masks must be a non-empty matrix')
    mask = action_masks & visited_masks.logical_not()
    mask = mask.clone()
    mask[:, 0] = False
    return mask


def _masked_log_softmax(logits, mask):
    masked = logits.masked_fill(mask.logical_not(), -float('inf'))
    return F.log_softmax(masked, dim=-1)


def factorized_action_log_probs(base_logits, frontier_logits, frontier_masks):
    """Preserve E0 STOP mass and redistribute only CONTINUE mass."""
    if (base_logits.shape != frontier_logits.shape or
            base_logits.shape != frontier_masks.shape):
        raise ValueError('Factorized action tensors are misaligned')
    if base_logits.ndim != 2 or base_logits.size(1) == 0:
        raise ValueError('Action logits must be a non-empty matrix')

    base_log_probs = F.log_softmax(base_logits, dim=-1)
    output = base_log_probs.clone()
    rows = frontier_masks.any(dim=-1)
    if not rows.any():
        return output

    masks = frontier_masks[rows]
    base_rows = base_log_probs[rows]
    continue_log_mass = torch.logsumexp(
        base_rows.masked_fill(masks.logical_not(), -float('inf')),
        dim=-1,
        keepdim=True,
    )
    conditional = _masked_log_softmax(frontier_logits[rows], masks)
    routed = continue_log_mass + conditional
    output[rows] = torch.where(masks, routed, base_rows)
    output[rows, 0] = base_rows[:, 0]
    return output


def factorized_greedy_actions(base_logits, frontier_logits, frontier_masks):
    """Keep E0's STOP decision; otherwise choose the routed frontier."""
    if (base_logits.shape != frontier_logits.shape or
            base_logits.shape != frontier_masks.shape):
        raise ValueError('Factorized action tensors are misaligned')
    base_actions = base_logits.argmax(dim=-1)
    has_frontier = frontier_masks.any(dim=-1)
    routed_actions = frontier_logits.masked_fill(
        frontier_masks.logical_not(), -float('inf')
    ).argmax(dim=-1)
    return torch.where(
        (base_actions == 0) | has_frontier.logical_not(),
        torch.zeros_like(base_actions),
        routed_actions,
    )


def conditional_frontier_kl(base_logits, frontier_logits, frontier_masks):
    """Mean KL(E15 || E0) inside the valid frontier set."""
    if (base_logits.shape != frontier_logits.shape or
            base_logits.shape != frontier_masks.shape):
        raise ValueError('Conditional KL tensors are misaligned')
    rows = frontier_masks.any(dim=-1)
    if not rows.any():
        return base_logits.new_zeros(())
    masks = frontier_masks[rows]
    base = _masked_log_softmax(base_logits[rows], masks)
    routed = _masked_log_softmax(frontier_logits[rows], masks)
    safe_base = torch.where(masks, base, torch.zeros_like(base))
    safe_routed = torch.where(masks, routed, torch.zeros_like(routed))
    terms = safe_routed.exp() * (safe_routed - safe_base)
    return terms.masked_select(masks).sum() / rows.sum()
