#!/usr/bin/env python3
"""Exercise E16 through the production model after a strict E0 load."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.smoke_e15_model import inputs


def conditional_probabilities(logits, masks):
    masked = logits.masked_fill(masks.logical_not(), -float('inf'))
    return torch.softmax(masked, dim=-1)


def probability_mismatch(log_probs, base_logits):
    """Return behavior-space error and a dtype-scaled FP tolerance."""
    expected = torch.softmax(base_logits, dim=-1)
    mismatch = (log_probs.exp() - expected).abs().max().item()
    tolerance = 8 * torch.finfo(log_probs.dtype).eps
    return mismatch, tolerance


def exercise_model(model):
    torch.manual_seed(16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.factorized_landmark.parameters():
        parameter.requires_grad_(True)
    nav_inputs = inputs(model.config.hidden_size)

    with torch.no_grad():
        enabled = model.forward_navigation(**nav_inputs)
        transport = model.factorized_landmark
        model.factorized_landmark = None
        disabled = model.forward_navigation(**{
            key: value for key, value in nav_inputs.items()
            if not key.startswith('gmap_transport_')
        })['global_logits']
        model.factorized_landmark = transport
    expected = torch.log_softmax(disabled, dim=-1)
    mismatch, tolerance = probability_mismatch(
        enabled['global_logits'], disabled
    )
    invalid_equal = torch.equal(
        torch.isneginf(enabled['global_logits']), torch.isneginf(expected)
    )
    action_mismatch = (
        enabled['factorized_greedy_actions'] != disabled.argmax(-1)
    ).sum().item()
    if mismatch > tolerance or not invalid_equal or action_mismatch:
        raise RuntimeError(
            'Zero-initialized E16 differs from E0: probability_mismatch=%g, '
            'tolerance=%g, invalid_masks_equal=%s, action_mismatch=%d' % (
                mismatch, tolerance, invalid_equal, action_mismatch
            )
        )
    if enabled['budget_activation_rate'].item() != 0:
        raise RuntimeError('Zero-initialized E16 activated its graph budget')

    with torch.no_grad():
        model.factorized_landmark.output.weight.normal_(std=0.05)
        model.factorized_landmark_budgeted = False
        routed = model.forward_navigation(**nav_inputs)
        frontier_masks = (
            nav_inputs['gmap_masks'] &
            nav_inputs['gmap_visited_masks'].logical_not()
        )
        frontier_masks[:, 0] = False
        base_probs = conditional_probabilities(
            routed['base_global_logits'], frontier_masks
        )
        routed_probs = conditional_probabilities(
            routed['global_logits'], frontier_masks
        )
        costs = (routed_probs > base_probs).to(
            nav_inputs['gmap_pair_dists'].dtype
        )
        current = nav_inputs['gmap_step_ids'].masked_fill(
            nav_inputs['gmap_visited_masks'].logical_not(), -1
        ).argmax(-1)
        for row in range(costs.size(0)):
            nav_inputs['gmap_pair_dists'][row, current[row]] = costs[row]
            nav_inputs['gmap_pair_dists'][row, :, current[row]] = costs[row]
        model.factorized_landmark_budgeted = True

    outputs = model.forward_navigation(**nav_inputs)
    if outputs['budget_activation_rate'].item() <= 0:
        raise RuntimeError('E16 smoke did not exercise the active projection')
    if (outputs['projected_route_cost'] >
            outputs['base_route_cost'] + 1e-6):
        raise RuntimeError('E16 exceeded the frozen E0 graph-cost budget')
    stop_mismatch = (
        outputs['global_logits'].exp()[:, 0] -
        torch.softmax(outputs['base_global_logits'], dim=-1)[:, 0]
    ).abs().max()
    if stop_mismatch.item() > 1e-7:
        raise RuntimeError('E16 changed the frozen E0 STOP probability')

    frontier = frontier_masks.nonzero(as_tuple=False)[0]
    outputs['global_logits'][frontier[0], frontier[1]].backward()
    if model.factorized_landmark.output.weight.grad.abs().sum() == 0:
        raise RuntimeError('Navigation objective did not reach E16 output')
    for name, parameter in model.named_parameters():
        if (not name.startswith('factorized_landmark.') and
                parameter.grad is not None):
            raise RuntimeError('E16 gradient reached frozen E0: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 590848:
        raise RuntimeError('E16 expected exactly 590,848 trainable parameters')
    return (trainable, mismatch, tolerance, action_mismatch,
            stop_mismatch.item(), outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.factorized_landmark_size', '128',
        'MODEL.factorized_landmark_budgeted', 'True',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    checkpoint = torch.load(args.baseline, map_location='cpu')
    weights = {
        key.replace('net.module.', 'net.').split('net.vln_bert.', 1)[1]: value
        for key, value in checkpoint['state_dict'].items()
        if key.replace('net.module.', 'net.').startswith('net.vln_bert.')
    }
    missing = model.load_state_dict(weights, strict=False)
    expected = {
        'factorized_landmark.' + key
        for key in model.factorized_landmark.state_dict()
    }
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError('Strict E0 model load mismatch: ' + str(missing))
    model.factorized_landmark.reset_output()
    del checkpoint, weights

    (trainable, mismatch, tolerance, action_mismatch,
     stop_mismatch, outputs) = exercise_model(model)
    print('E16_MODEL_SMOKE_PASSED trainable=%d '
          'max_probability_mismatch=%g tolerance=%g action_mismatch=%d '
          'stop_probability_mismatch=%g '
          'budget_activation_rate=%g projected_cost=%g base_cost=%g '
          'frozen_gradient=0' % (
              trainable, mismatch, tolerance, action_mismatch, stop_mismatch,
              outputs['budget_activation_rate'].item(),
              outputs['projected_route_cost'].item(),
              outputs['base_route_cost'].item(),
          ))


if __name__ == '__main__':
    main()
