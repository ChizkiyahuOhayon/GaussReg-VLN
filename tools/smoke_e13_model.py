#!/usr/bin/env python3
"""Exercise E13 through the production model after a strict E0 load."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def exercise_model(model):
    torch.manual_seed(13)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.instruction_coverage.parameters():
        parameter.requires_grad_(True)

    width = model.config.hidden_size
    txt_embeds = torch.randn(2, 6, width)
    txt_masks = torch.tensor([
        [True, True, True, True, True, False],
        [True, True, True, True, False, False],
    ])
    view_embeds = torch.randn(2, 5, width)
    view_masks = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, False, False],
    ])
    view_evidence, _ = model.forward_instruction_evidence(
        view_embeds, view_masks, txt_embeds, txt_masks
    )
    graph_evidence = torch.cat([
        torch.zeros(2, 1, 4),
        view_evidence[:, :4],
    ], dim=1)
    inputs = (
        txt_embeds, txt_masks,
        [[None, '0', '1', 'g0', 'g1'], [None, '0', '1', 'g0', None]],
        torch.tensor([[0, 1, 2, 0, 0]] * 2),
        torch.randn(2, 5, width), torch.randn(2, 5, 7),
        torch.tensor([[True] * 5, [True, True, True, True, False]]),
        torch.tensor([[False, True, True, False, False]] * 2),
        torch.zeros(2, 5, 5), torch.ones(2, 5, dtype=torch.long),
    )

    with torch.no_grad():
        enabled = model.forward_navigation(
            *inputs, gmap_instruction_evidence=graph_evidence
        )
        coverage = model.instruction_coverage
        model.instruction_coverage = None
        disabled = model.forward_navigation(*inputs)['global_logits']
        model.instruction_coverage = coverage
    if not torch.equal(enabled['global_logits'], disabled):
        raise RuntimeError('Zero-initialized E13 is not exactly E0')
    if not torch.equal(enabled['base_global_logits'], disabled):
        raise RuntimeError('E13 base logits differ from E0')
    if graph_evidence.requires_grad or view_evidence.requires_grad:
        raise RuntimeError('Frozen evidence is attached to autograd')
    finite = torch.isfinite(disabled)
    max_logit_mismatch = (
        enabled['global_logits'][finite] - disabled[finite]
    ).abs().max().item()
    action_mismatch = (
        enabled['global_logits'].argmax(dim=-1) != disabled.argmax(dim=-1)
    ).sum().item()

    outputs = model.forward_navigation(
        *inputs, gmap_instruction_evidence=graph_evidence
    )
    objective = torch.nn.functional.log_softmax(
        outputs['global_logits'], dim=1
    )[:, 3].sum()
    objective.backward()
    for name, parameter in model.named_parameters():
        if 'instruction_coverage.' not in name and parameter.grad is not None:
            raise RuntimeError('E13 gradient reached frozen E0: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)
    if model.instruction_coverage.output.weight.grad.abs().sum() == 0:
        raise RuntimeError('Navigation objective did not reach E13 residual')

    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable != 449:
        raise RuntimeError('E13 expected exactly 449 trainable parameters')
    return trainable, max_logit_mismatch, action_mismatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.instruction_coverage_hidden_size', '32',
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
        'instruction_coverage.' + key
        for key in model.instruction_coverage.state_dict()
    }
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError('Strict E0 model load mismatch: ' + str(missing))
    model.instruction_coverage.reset_output()
    del checkpoint, weights

    trainable, mismatch, action_mismatch = exercise_model(model)
    print('E13_MODEL_SMOKE_PASSED trainable=%d max_logit_mismatch=%g '
          'action_mismatch=%d evidence_detached=1 frozen_gradient=0' % (
              trainable, mismatch, action_mismatch
          ))


if __name__ == '__main__':
    main()
