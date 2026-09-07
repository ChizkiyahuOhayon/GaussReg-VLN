#!/usr/bin/env python3
"""Exercise the real navigation forward; production mode also loads strict E0."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exercise_model(model):
    from vlnce_baselines.successor import successor_loss

    torch.manual_seed(12)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.successor.parameters():
        parameter.requires_grad_(True)
    width = model.config.hidden_size
    inputs = (
        torch.randn(2, 4, width),
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
        [[None, '0', '1', 'g0', 'g1'], [None, '0', '1', 'g0', None]],
        torch.tensor([[0, 1, 2, 0, 0]] * 2),
        torch.randn(2, 5, width), torch.randn(2, 5, 7),
        torch.tensor([[True] * 5, [True, True, True, True, False]]),
        torch.tensor([[False, True, True, False, False]] * 2),
        torch.zeros(2, 5, 5), torch.ones(2, 5, dtype=torch.long),
    )
    with torch.no_grad():
        model.successor_sampling_base = True
        bypassed = model.forward_navigation(*inputs)['global_logits']
        model.successor_sampling_base = False
        decoder = model.successor
        model.successor = None
        disabled = model.forward_navigation(*inputs)['global_logits']
        model.successor = decoder
    if not torch.equal(bypassed, disabled):
        raise RuntimeError('E12 behavior sampling is not identical to disabled E0')

    outputs = model.forward_navigation(*inputs)
    logits = outputs['global_logits']
    if logits.shape != (2, 5) or not torch.equal(torch.isfinite(logits), torch.isfinite(disabled)):
        raise RuntimeError('Virtual evidence leaked into action slots or changed action masks')
    with torch.no_grad():
        measured = outputs['successor_features'].detach().clone()
        measured[:, 3] = torch.randn(2, width)
        target = model.forward_navigation(*inputs, successor_override=measured)
    feature, planner = successor_loss(
        outputs['successor_features'][:, 3], measured[:, 3],
        logits, target['global_logits'],
    )
    (feature + planner).backward()
    for name, parameter in model.named_parameters():
        if 'successor.' not in name and parameter.grad is not None:
            raise RuntimeError('E12 gradient reached frozen E0: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)
    if model.successor.output.weight.grad.abs().sum() == 0:
        raise RuntimeError('Arrival supervision did not reach successor decoder')
    if model.successor.attention.in_proj_weight.grad.abs().sum() == 0:
        raise RuntimeError('Arrival supervision did not reach cross-attention')
    model.zero_grad()
    # Independently prove that the virtual nodes affect the frozen action readout.
    outputs = model.forward_navigation(*inputs)
    action_objective = torch.nn.functional.log_softmax(outputs['global_logits'], 1)[:, 3].sum()
    action_objective.backward()
    if model.successor.output.weight.grad.abs().sum() == 0:
        raise RuntimeError('Prospective evidence does not influence navigation')
    model.zero_grad()
    return sum(p.numel() for p in model.successor.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.successor_hidden_size', '256',
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
    expected = {'successor.' + key for key in model.successor.state_dict()}
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError('Strict E0 model load mismatch: ' + str(missing))
    del checkpoint, weights
    count = exercise_model(model)
    print('E12_MODEL_SMOKE_PASSED trainable=%d E0_bypass_equal=1 '
          'virtual_action_count=0 frozen_gradient=0' % count)


if __name__ == '__main__':
    main()
