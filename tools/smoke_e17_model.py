#!/usr/bin/env python3
"""Exercise E17 through the production model after a strict E0 load."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.smoke_e15_model import inputs
from tools.smoke_e16_model import probability_mismatch


def exercise_model(model):
    torch.manual_seed(17)
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
    mismatch, tolerance = probability_mismatch(
        enabled['global_logits'], disabled
    )
    invalid_equal = torch.equal(
        torch.isneginf(enabled['global_logits']),
        torch.isneginf(torch.log_softmax(disabled, dim=-1)),
    )
    action_mismatch = (
        enabled['factorized_greedy_actions'] != disabled.argmax(-1)
    ).sum().item()
    if mismatch > tolerance or not invalid_equal or action_mismatch:
        raise RuntimeError(
            'Zero-initialized E17 differs from E0: probability_mismatch=%g, '
            'tolerance=%g, invalid_masks_equal=%s, action_mismatch=%d' % (
                mismatch, tolerance, invalid_equal, action_mismatch
            )
        )
    progress = enabled['instruction_stage_progress']
    if (not torch.isfinite(progress).all() or
            not torch.allclose(progress.sum(-1), torch.ones_like(
                progress[:, 0]
            ))):
        raise RuntimeError('E17 stage posterior is invalid')

    with torch.no_grad():
        model.factorized_landmark.output.weight.normal_(std=0.05)
    outputs = model.forward_navigation(**nav_inputs)
    stop_mismatch = (
        outputs['global_logits'].exp()[:, 0] -
        torch.softmax(outputs['base_global_logits'], dim=-1)[:, 0]
    ).abs().max().item()
    if stop_mismatch > 1e-7:
        raise RuntimeError('E17 changed the frozen E0 STOP probability')
    if any(not torch.isfinite(outputs[name]) for name in [
            'instruction_stage_expected', 'instruction_stage_entropy',
            'instruction_stage_advance_mass']):
        raise RuntimeError('E17 stage diagnostics are nonfinite')

    frontier_masks = (
        nav_inputs['gmap_masks'] &
        nav_inputs['gmap_visited_masks'].logical_not()
    )
    frontier_masks[:, 0] = False
    frontier = frontier_masks.nonzero(as_tuple=False)[0]
    outputs['global_logits'][frontier[0], frontier[1]].backward()
    required = ('view_projection.weight', 'text_projection.weight',
                'output.weight')
    parameters = dict(model.factorized_landmark.named_parameters())
    if any(parameters[name].grad is None or
           parameters[name].grad.abs().sum() == 0 for name in required):
        raise RuntimeError('Navigation objective did not reach all E17 layers')
    for name, parameter in model.named_parameters():
        if (not name.startswith('factorized_landmark.') and
                parameter.grad is not None):
            raise RuntimeError('E17 gradient reached frozen E0: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 590848:
        raise RuntimeError('E17 expected exactly 590,848 trainable parameters')
    return (trainable, mismatch, tolerance, action_mismatch,
            stop_mismatch, outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.factorized_landmark_size', '128',
        'MODEL.factorized_landmark_budgeted', 'False',
        'MODEL.factorized_landmark_monotonic', 'True',
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
    print('E17_MODEL_SMOKE_PASSED trainable=%d '
          'max_probability_mismatch=%g tolerance=%g action_mismatch=%d '
          'stop_probability_mismatch=%g stage_expected=%g '
          'stage_entropy=%g stage_advance_mass=%g frozen_gradient=0' % (
              trainable, mismatch, tolerance, action_mismatch, stop_mismatch,
              outputs['instruction_stage_expected'].item(),
              outputs['instruction_stage_entropy'].item(),
              outputs['instruction_stage_advance_mass'].item(),
          ))


if __name__ == '__main__':
    main()
