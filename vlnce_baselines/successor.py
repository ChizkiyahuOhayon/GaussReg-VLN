"""Branch-linked future evidence; no simulator, goal or reward dependency."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def decoder_digest(state):
    """Fingerprint learned floating tensors, excluding the update counter."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if value.is_floating_point():
            digest.update(name.encode('utf-8'))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SuccessorDecoder(nn.Module):
    """Predict an arrival panorama from a frontier, instruction and current graph."""

    def __init__(self, feature_size, hidden_size=256):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_size)
        self.graph_projection = nn.Linear(feature_size, hidden_size)
        self.text_projection = nn.Linear(feature_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, 4, dropout=0.0)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, feature_size)
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))

    def forward(self, graph, graph_mask, text, text_mask):
        # E0 produces a stationary feature space for both inputs and targets.
        query = self.graph_projection(self.input_norm(graph.detach()))
        language = self.text_projection(self.input_norm(text.detach()))
        memory = torch.cat([language, query], dim=1)
        mask = torch.cat([text_mask, graph_mask], dim=1)
        attended = self.attention(
            query.transpose(0, 1), memory.transpose(0, 1),
            memory.transpose(0, 1), key_padding_mask=~mask,
            need_weights=False,
        )[0].transpose(0, 1)
        hidden = self.attention_norm(query + attended)
        return self.output(self.output_norm(hidden + self.ffn(hidden)))


def expand_successor_graph(graph, future, position, valid, visited, distances):
    """Append non-actionable evidence, with each virtual slot owned by slot i.

    Distances are known owner distances, not imagined free-space connectivity.
    The caller reads actions only from the first N slots.
    """
    future_mask = valid & ~visited
    future_mask = future_mask.clone()
    future_mask[:, 0] = False
    features = torch.cat([graph, future + position], dim=1)
    mask = torch.cat([valid, future_mask], dim=1)
    expanded_distances = distances.repeat(1, 2, 2)
    return features, mask, expanded_distances


def arrival_pairs(steps, tolerance=0.5):
    """Return (source step, source row, action, measured arrival feature).

    Only adjacent real decisions of the same environment can form a pair. The
    source action must have been executed and its commanded location reached.
    """
    pairs = []
    for step_index, (source, following) in enumerate(zip(steps, steps[1:])):
        next_rows = {index: row for row, index in enumerate(following['indices'])}
        for row, identity in enumerate(source['indices']):
            action = int(source['action'][row])
            if not source['executed_move'][row] or action == 0:
                continue
            if identity not in next_rows:
                continue
            next_row = next_rows[identity]
            distance = torch.norm(
                following['current_position'][next_row] -
                source['successor_position'][row]
            )
            if not torch.isfinite(distance) or distance > tolerance:
                continue
            nav = following['input']
            visited = nav['gmap_visited_masks'][next_row] & nav['gmap_masks'][next_row]
            current = nav['gmap_step_ids'][next_row].masked_fill(~visited, -1).argmax()
            target = nav['gmap_img_fts'][next_row, current].detach()
            pairs.append((step_index, row, action, target))
    return pairs


def successor_loss(prediction, arrival, student_logits, teacher_logits):
    """Arrival fidelity plus decision consistency under a measured substitution."""
    if not torch.isfinite(prediction).all() or not torch.isfinite(arrival).all():
        raise ValueError('successor features must be finite')
    valid = torch.isfinite(student_logits)
    if not torch.equal(valid, torch.isfinite(teacher_logits)) or not valid.any(1).all():
        raise ValueError('student and target action masks must match and be nonempty')
    if torch.isnan(student_logits).any() or torch.isnan(teacher_logits).any():
        raise ValueError('action logits must be finite or masked negative infinity')
    if torch.isposinf(student_logits).any() or torch.isposinf(teacher_logits).any():
        raise ValueError('action logits must be finite or masked negative infinity')
    feature_loss = F.smooth_l1_loss(prediction, arrival.detach())
    # Replacing masked log probabilities before arithmetic avoids 0 * inf NaNs.
    student_logp = F.log_softmax(student_logits, dim=1).masked_fill(~valid, 0)
    teacher_logp = F.log_softmax(teacher_logits.detach(), dim=1).masked_fill(~valid, 0)
    teacher_prob = F.softmax(teacher_logits.detach(), dim=1).masked_fill(~valid, 0)
    planner_loss = (teacher_prob * (teacher_logp - student_logp)).sum(1).mean()
    return feature_loss, planner_loss


def update_successor(trainer):
    """One supervised update; the GRPO buffer is only a rollout transport here."""
    examples = [
        (sample, pair)
        for sample in trainer.data_buffer
        for pair in arrival_pairs(sample['data_buffer'], trainer.config.GRPO.loc_noise)
    ]
    trainer.logs['successor_pairs'].append(len(examples))
    trainer.optimizer.zero_grad()
    if not examples:
        trainer.data_buffer.clear()
        trainer.scheduler.step()
        return

    trainer.set_policy_mode('train')
    feature_total, planner_total, copy_total = 0., 0., 0.
    for sample, (step_index, row, action, arrival) in examples:
        step = sample['data_buffer'][step_index]
        nav = {}
        for key, value in step['input'].items():
            if isinstance(value, torch.Tensor):
                nav[key] = value[row:row + 1].to(trainer.device)
            elif isinstance(value, list):
                nav[key] = [value[row]]
            else:
                nav[key] = value
        identity = step['indices'][row]
        nav['txt_embeds'] = sample['initial_txt_embeds'][identity:identity + 1].to(trainer.device)
        nav['txt_masks'] = sample['initial_txt_masks'][identity:identity + 1].to(trainer.device)
        outputs = trainer.policy.net(**nav)
        arrival = arrival.unsqueeze(0).to(trainer.device)
        predicted = outputs['successor_features'][:, action]
        with torch.no_grad():
            measured = outputs['successor_features'].detach().clone()
            measured[:, action] = arrival
            target = trainer.policy.net(**nav, successor_override=measured)
            copy_total += F.smooth_l1_loss(nav['gmap_img_fts'][:, action], arrival).item()
        feature, planner = successor_loss(
            predicted, arrival, outputs['global_logits'], target['global_logits']
        )
        loss = feature + planner
        if not torch.isfinite(loss):
            raise RuntimeError('E12 loss is nonfinite')
        (loss / len(examples)).backward()
        feature_total += feature.item()
        planner_total += planner.item()

    decoder = trainer.policy.net.vln_bert.successor
    parameters = list(decoder.parameters())
    norm = torch.nn.utils.clip_grad_norm_(parameters, trainer.max_grad_norm)
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError('E12 requires a finite nonzero decoder gradient')
    trainer.optimizer.step()
    decoder.updates.add_(1)
    trainer.scheduler.step()
    trainer.logs['grad_norm'].append(norm.item())
    trainer.logs['successor_feature_loss'].append(feature_total / len(examples))
    trainer.logs['successor_planner_kl'].append(planner_total / len(examples))
    trainer.logs['successor_copy_loss'].append(copy_total / len(examples))
    trainer.logs['total_loss'].append((feature_total + planner_total) / len(examples))
    trainer.optimizer.zero_grad()
    trainer.data_buffer.clear()
