"""Monotonic instruction-stage routing from real graph observations."""

import math

import torch
from torch.nn import functional as F

from vlnce_baselines.landmark_transport import (
    LandmarkTransport,
    _masked_softmax,
    ordered_slot_content,
)


def monotonic_progress(emissions, history_masks, slot_masks):
    """Align an ordered observation history to ordered instruction slots."""
    if emissions.ndim != 3:
        raise ValueError('Emissions must have shape [batch, history, slots]')
    if (history_masks.shape != emissions.shape[:2] or
            slot_masks.shape != (emissions.size(0), emissions.size(2))):
        raise ValueError('Monotonic alignment tensors are misaligned')
    if history_masks.dtype != torch.bool or slot_masks.dtype != torch.bool:
        raise ValueError('Monotonic alignment masks must be boolean')

    rows = []
    for row in range(emissions.size(0)):
        slot_indices = slot_masks[row].nonzero(as_tuple=False).squeeze(-1)
        if slot_indices.numel() == 0:
            raise ValueError('Each instruction needs at least one valid slot')
        history_indices = history_masks[row].nonzero(
            as_tuple=False
        ).squeeze(-1)
        if history_indices.numel() == 0:
            probabilities = emissions.new_zeros(emissions.size(2))
            probabilities[slot_indices[0]] = 1
            rows.append(probabilities)
            continue

        values = emissions[row].index_select(
            0, history_indices
        ).index_select(1, slot_indices)
        if not torch.isfinite(values).all():
            raise ValueError('Valid monotonic emissions must be finite')
        alpha = values.new_full((values.size(1),), -float('inf'))
        alpha[0] = values[0, 0]
        for observation in values[1:]:
            advance = torch.cat([
                alpha.new_full((1,), -float('inf')), alpha[:-1]
            ])
            alpha = observation + torch.logsumexp(
                torch.stack([alpha, advance]), dim=0
            )
        probabilities = emissions.new_zeros(emissions.size(2))
        probabilities = probabilities.scatter(
            0, slot_indices, torch.softmax(alpha, dim=0)
        )
        rows.append(probabilities)
    return torch.stack(rows)


def ordered_history(emissions, history_masks, step_ids):
    """Sort node emissions by visit step while keeping padding masked."""
    if (emissions.ndim != 3 or
            history_masks.shape != emissions.shape[:2] or
            step_ids.shape != history_masks.shape):
        raise ValueError('Graph history tensors are misaligned')
    if history_masks.dtype != torch.bool or step_ids.is_floating_point():
        raise ValueError('History mask must be boolean and step ids integral')
    if (step_ids.masked_select(history_masks) < 0).any():
        raise ValueError('Valid history step ids must be nonnegative')

    node_ids = torch.arange(
        step_ids.size(1), device=step_ids.device, dtype=step_ids.dtype
    ).unsqueeze(0)
    keys = step_ids * (step_ids.size(1) + 1) + node_ids
    keys = keys.masked_fill(
        history_masks.logical_not(), torch.iinfo(step_ids.dtype).max
    )
    order = keys.argsort(dim=1)
    ordered_emissions = emissions.gather(
        1, order.unsqueeze(-1).expand_as(emissions)
    )
    ordered_masks = history_masks.gather(1, order)
    return ordered_emissions, ordered_masks


class MonotonicLandmarkTransport(LandmarkTransport):
    """Condition E15 landmark memory on monotonic real-history progress."""

    def forward(self, view_embeds, view_masks, text_embeds, text_masks,
                step_ids, visited_masks):
        if view_embeds.shape[:3] != view_masks.shape:
            raise ValueError('Transport views and masks are misaligned')
        if (view_embeds.size(0) != text_embeds.size(0) or
                step_ids.shape != view_masks.shape[:2] or
                visited_masks.shape != step_ids.shape):
            raise ValueError('Monotonic transport tensors are misaligned')

        slots, slot_masks = ordered_slot_content(
            text_embeds.detach(), text_masks
        )
        views = F.gelu(self.view_projection(view_embeds.detach()))
        slots = F.gelu(self.text_projection(slots))
        compatibility = torch.einsum('bnsh,bkh->bnsk', views, slots)
        compatibility = compatibility / math.sqrt(self.transport_size)

        source_mask = view_masks.unsqueeze(-1).expand_as(compatibility)
        source_count = view_masks.sum(dim=2).clamp_min(1)
        node_emissions = torch.logsumexp(
            compatibility.masked_fill(
                source_mask.logical_not(), -float('inf')
            ),
            dim=2,
        ) - source_count.log().unsqueeze(-1).to(compatibility.dtype)
        history_masks = visited_masks & view_masks.any(dim=-1)
        history_emissions, history_masks = ordered_history(
            node_emissions, history_masks, step_ids
        )
        progress = monotonic_progress(
            history_emissions, history_masks, slot_masks
        )

        slot_mask = slot_masks[:, None, None, :].expand_as(compatibility)
        slot_weights = _masked_softmax(compatibility, slot_mask, dim=-1)
        source_weights = _masked_softmax(
            compatibility, source_mask, dim=2
        )
        memory = torch.einsum(
            'bnsk,bnsk,bnsh,bkh->bnkh',
            slot_weights, source_weights, views, slots,
        )
        memory = memory * progress[:, None, :, None]
        residual = self.output(memory.flatten(start_dim=2))
        node_masks = view_masks.any(dim=-1)
        residual = residual * node_masks.unsqueeze(-1).to(residual.dtype)

        entropy_terms = -(slot_weights.clamp_min(1e-9).log() * slot_weights)
        valid_sources = source_mask & slot_mask
        entropy = entropy_terms.masked_select(valid_sources).sum()
        entropy = entropy / valid_sources.sum().clamp_min(1)
        stage_positions = torch.arange(
            progress.size(1), device=progress.device, dtype=progress.dtype
        )
        first_slots = slot_masks.to(torch.long).argmax(dim=-1)
        stage_entropy = -(
            progress.clamp_min(1e-9).log() * progress
        ).sum(dim=-1).mean()
        diagnostics = {
            'entropy': entropy.detach(),
            'residual_norm': residual.detach().norm(dim=-1)[node_masks].mean()
            if node_masks.any() else residual.new_zeros(()),
            'stage_progress': progress.detach(),
            'stage_expected': (
                progress * stage_positions
            ).sum(dim=-1).mean().detach(),
            'stage_entropy': stage_entropy.detach(),
            'stage_advance_mass': (
                1 - progress.gather(1, first_slots.unsqueeze(1)).squeeze(1)
            ).mean().detach(),
        }
        return residual, diagnostics
