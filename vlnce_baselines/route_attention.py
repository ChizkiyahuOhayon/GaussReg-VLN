"""Executable-route-conditioned final graph aggregation (E18)."""

from copy import deepcopy

import torch
from torch import nn

from vlnce_baselines.common.ops import extend_neg_masks


def batch_executable_route_masks(gmaps, current_vps, vp_ids, max_nodes, device):
    """Use the same front and visited-node path as the control backtracker."""
    if not (len(gmaps) == len(current_vps) == len(vp_ids)):
        raise ValueError('Misaligned route batch')
    masks = torch.zeros(len(gmaps), max_nodes, max_nodes, dtype=torch.bool)
    for batch, (graph, current, ids) in enumerate(zip(gmaps, current_vps, vp_ids)):
        if not ids or ids[0] is not None or len(ids) > max_nodes:
            raise ValueError('Invalid route graph slots')
        indices = {vp: index for index, vp in enumerate(ids)}
        masks[batch, :, :len(ids)] = True
        for ghost in graph.ghost_pos:
            if ghost not in indices:
                raise ValueError('Missing frontier in route graph')
            _, front = graph.front_to_ghost_dist(ghost)
            route = graph.shortest_path[current][front] + [ghost]
            if current not in route or any(vp not in indices for vp in route):
                raise ValueError('Controller route contains an unknown graph node')
            row = indices[ghost]
            masks[batch, row] = False
            masks[batch, row, [indices[vp] for vp in route]] = True
    return masks.to(device=device)


class ExecutableRouteAttention(nn.Module):
    """A pretrained final planning layer and readout, not a logit residual."""

    def __init__(self, model, full_graph=False):
        super().__init__()
        self.full_graph = full_graph
        for name, module in self.e0_modules(model).items():
            self.add_module(name, deepcopy(module))
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    @staticmethod
    def e0_modules(model):
        return {
            'layer': model.global_encoder.encoder.x_layers[-1],
            'query': model.graph_query_text,
            'transform': model.graph_attentioned_txt_embeds_transform,
            'head': model.global_sap_head,
        }

    def copy_from_e0(self, model):
        # In-place load preserves the optimizer's parameter references.
        for name, module in self.e0_modules(model).items():
            getattr(self, name).load_state_dict(module.state_dict(), strict=True)

    def forward(self, text, text_masks, graph, graph_masks, route_masks, spatial):
        batch, nodes = graph.shape[:2]
        if route_masks is None or route_masks.shape != (batch, nodes, nodes):
            raise ValueError('E18 route mask shape must be [batch, nodes, nodes]')
        if route_masks.dtype != torch.bool:
            raise ValueError('E18 route mask must be boolean')
        if (not text_masks.any(-1).all() or not graph_masks[:, 0].all() or
                not (route_masks & graph_masks[:, None, :]).any(-1).all()):
            raise ValueError('E18 masks must provide text, STOP and attention keys')
        bias = spatial.detach() if spatial is not None else 0
        if not self.full_graph:
            bias = bias + (~route_masks[:, None]).to(graph.dtype) * -10000.0
        text_mask = extend_neg_masks(text_masks)
        text, graph = self.layer(
            text.detach(), text_mask, graph.detach(),
            extend_neg_masks(graph_masks), graph_sprels=bias,
        )
        attended, _ = self.query(graph, text, attention_mask=text_mask)
        return self.head(torch.cat([graph, self.transform(attended)], -1)).squeeze(-1)


def validate_training_config(config):
    """One fixed recipe shared by the route and equal-capacity control arms."""
    grpo = config.GRPO
    expected = dict(
        route_attention_only=True, sample_num=8, update_epochs=1,
        batch_size=1, lr=2e-5, min_lr_ratio=.25, warmup_iters=0,
        grpo_beta=.04, max_grad_norm=2., load_from_ckpt=True,
        back_algo='control', is_requeue=False, enable_amp=False,
        enable_all_dropouts=False, dropout_in_sampling=False, waypoint_aug=False,
    )
    excluded = (
        'gauss_only', 'candidate_scorer_only', 'gaussian_bev_only',
        'anchor_repair_only', 'hindsight_stop_only', 'terminal_commit_only',
        'success_set_commit', 'setwise_group_policy', 'frontier_advantage',
        'geo_token_only', 'successor_only', 'instruction_coverage_only',
        'landmark_transport_only', 'factorized_landmark_only',
        'budgeted_factorized_landmark_only', 'monotonic_factorized_landmark_only',
    )
    if (any(getattr(grpo, key, None) != value for key, value in expected.items()) or
            any(getattr(grpo, key, False) for key in excluded) or
            not config.MODEL.route_attention or config.MODEL.task_type != 'r2r' or
            config.GPU_NUMBERS != 1 or config.NUM_ENVIRONMENTS != 1 or
            config.TASK_CONFIG.DATASET.SPLIT != 'train' or
            config.TASK_CONFIG.DATASET.SUFFIX != '_10'):
        raise ValueError('E18 requires the fixed independent E0 training recipe')
