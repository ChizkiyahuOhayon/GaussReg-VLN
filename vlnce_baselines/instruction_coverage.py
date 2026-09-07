import torch
from torch import nn
from torch.nn.functional import gelu


NUM_INSTRUCTION_SLOTS = 4


def ordered_slot_assignments(token_masks, num_slots=NUM_INSTRUCTION_SLOTS):
    """Assign each valid token to one contiguous instruction slot."""
    if token_masks.ndim != 2:
        raise ValueError('token_masks must have shape [batch, tokens]')
    if num_slots <= 0:
        raise ValueError('num_slots must be positive')
    lengths = token_masks.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError('each instruction must contain at least one token')

    ranks = token_masks.long().cumsum(dim=1) - 1
    slot_ids = torch.div(
        ranks * num_slots,
        lengths.unsqueeze(1),
        rounding_mode='floor',
    ).clamp_(0, num_slots - 1)
    assignments = torch.nn.functional.one_hot(
        slot_ids, num_classes=num_slots
    ).to(torch.bool)
    assignments &= token_masks.unsqueeze(-1)
    return assignments, assignments.any(dim=1)


def instruction_slot_evidence(attention_scores, token_masks):
    """Aggregate graph-to-text attention into ordered instruction slots."""
    if attention_scores.ndim != 4:
        raise ValueError(
            'attention_scores must have shape [batch, heads, views, tokens]'
        )
    if attention_scores.shape[0] != token_masks.shape[0] or \
            attention_scores.shape[-1] != token_masks.shape[1]:
        raise ValueError('attention scores and token masks are misaligned')

    assignments, slot_masks = ordered_slot_assignments(token_masks)
    scores = attention_scores.masked_fill(
        token_masks[:, None, None, :].logical_not(), -float('inf')
    )
    probabilities = torch.softmax(scores, dim=-1).mean(dim=1)
    evidence = torch.einsum(
        'bvt,btk->bvk', probabilities, assignments.to(probabilities.dtype)
    )
    return evidence, slot_masks


def aggregate_view_evidence(view_evidence, view_masks):
    """Pool real panorama views with an idempotent per-slot maximum."""
    if view_evidence.ndim != 3 or view_masks.shape != view_evidence.shape[:2]:
        raise ValueError('view evidence and masks are misaligned')
    if torch.any(view_masks.sum(dim=1) == 0):
        raise ValueError('each panorama must contain at least one valid view')
    masked = view_evidence.masked_fill(
        view_masks.unsqueeze(-1).logical_not(), -float('inf')
    )
    return masked.max(dim=1).values


def coverage_features(evidence, valid_masks, visited_masks):
    """Compute real-history coverage and per-node marginal evidence."""
    if evidence.ndim != 3 or valid_masks.shape != evidence.shape[:2] or \
            visited_masks.shape != evidence.shape[:2]:
        raise ValueError('graph evidence and masks are misaligned')
    visited = valid_masks & visited_masks
    visited = visited.clone()
    visited[:, 0] = False
    history = evidence.masked_fill(
        visited.unsqueeze(-1).logical_not(), 0.0
    )
    coverage = history.max(dim=1).values
    marginal = evidence * (1.0 - coverage.unsqueeze(1))
    return coverage, marginal


class InstructionCoverageResidual(nn.Module):
    """Score current frontiers by evidence not covered by real history."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden = nn.Linear(NUM_INSTRUCTION_SLOTS * 3, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, evidence, valid_masks, visited_masks):
        evidence = evidence.detach()
        coverage, marginal = coverage_features(
            evidence, valid_masks, visited_masks
        )
        inputs = torch.cat([
            evidence,
            coverage.unsqueeze(1).expand_as(evidence),
            marginal,
        ], dim=-1)
        residual = self.output(gelu(self.hidden(inputs))).squeeze(-1)
        frontier_masks = valid_masks & visited_masks.logical_not()
        frontier_masks = frontier_masks.clone()
        frontier_masks[:, 0] = False
        residual = residual.masked_fill(frontier_masks.logical_not(), 0.0)
        return residual, coverage, marginal
