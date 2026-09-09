"""Instruction-conditioned transport from real observations to graph tokens."""

import math

import torch
from torch import nn
from torch.nn import functional as F


NUM_LANDMARK_SLOTS = 4


def ordered_slot_content(text_embeds, text_masks, num_slots=NUM_LANDMARK_SLOTS):
    """Mean-pool valid instruction tokens into ordered contiguous slots."""
    if text_embeds.shape[:2] != text_masks.shape:
        raise ValueError('Text embeddings and masks are misaligned')
    lengths = text_masks.long().sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError('Each instruction must contain at least one token')

    positions = torch.arange(
        text_masks.size(1), device=text_masks.device
    ).unsqueeze(0)
    slot_ids = positions * num_slots // lengths.unsqueeze(1)
    slot_ids = slot_ids.clamp_max(num_slots - 1)
    assignments = F.one_hot(slot_ids, num_slots).bool()
    assignments = assignments & text_masks.unsqueeze(-1)
    counts = assignments.sum(dim=1)
    slots = torch.einsum(
        'btk,bth->bkh', assignments.to(text_embeds.dtype), text_embeds
    ) / counts.clamp_min(1).unsqueeze(-1).to(text_embeds.dtype)
    return slots, counts > 0


def _masked_softmax(logits, mask, dim):
    masked = logits.masked_fill(mask.logical_not(), -torch.finfo(logits.dtype).max)
    weights = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)


class LandmarkTransport(nn.Module):
    """Write content-level instruction evidence into graph representations."""

    def __init__(self, hidden_size, transport_size):
        super().__init__()
        if transport_size <= 0:
            raise ValueError('transport_size must be positive')
        self.transport_size = transport_size
        self.view_projection = nn.Linear(hidden_size, transport_size)
        self.text_projection = nn.Linear(hidden_size, transport_size)
        self.output = nn.Linear(
            NUM_LANDMARK_SLOTS * transport_size, hidden_size
        )
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, view_embeds, view_masks, text_embeds, text_masks):
        if view_embeds.shape[:3] != view_masks.shape:
            raise ValueError('Transport views and masks are misaligned')
        if view_embeds.size(0) != text_embeds.size(0):
            raise ValueError('Transport views and text have different batches')

        slots, slot_masks = ordered_slot_content(
            text_embeds.detach(), text_masks
        )
        views = F.gelu(self.view_projection(view_embeds.detach()))
        slots = F.gelu(self.text_projection(slots))
        compatibility = torch.einsum('bnsh,bkh->bnsk', views, slots)
        compatibility = compatibility / math.sqrt(self.transport_size)

        slot_mask = slot_masks[:, None, None, :].expand_as(compatibility)
        slot_weights = _masked_softmax(compatibility, slot_mask, dim=-1)
        source_mask = view_masks.unsqueeze(-1).expand_as(compatibility)
        source_weights = _masked_softmax(compatibility, source_mask, dim=2)
        memory = torch.einsum(
            'bnsk,bnsk,bnsh,bkh->bnkh',
            slot_weights, source_weights, views, slots,
        )
        residual = self.output(memory.flatten(start_dim=2))
        node_masks = view_masks.any(dim=-1)
        residual = residual * node_masks.unsqueeze(-1).to(residual.dtype)

        entropy_terms = -(slot_weights.clamp_min(1e-9).log() * slot_weights)
        valid_sources = view_masks.unsqueeze(-1) & slot_mask
        entropy = entropy_terms.masked_select(valid_sources).sum()
        entropy = entropy / valid_sources.sum().clamp_min(1)
        diagnostics = {
            'entropy': entropy.detach(),
            'residual_norm': residual.detach().norm(dim=-1)[node_masks].mean()
            if node_masks.any() else residual.new_zeros(()),
        }
        return residual, diagnostics
