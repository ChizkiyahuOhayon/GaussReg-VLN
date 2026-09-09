#!/usr/bin/env python3
"""Exercise E15 through the production model after a strict E0 load."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inputs(width):
    view_masks = torch.tensor([
        [[False, False, False], [True, True, True], [True, False, False],
         [True, True, False], [True, False, False]],
        [[False, False, False], [True, True, False], [True, False, False],
         [True, False, False], [False, False, False]],
    ])
    return dict(
        txt_embeds=torch.randn(2, 8, width),
        txt_masks=torch.tensor([
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False, False],
        ]),
        gmap_vpids=[[None, '0', '1', 'g0', 'g1']] * 2,
        gmap_step_ids=torch.tensor([[0, 1, 2, 0, 0]] * 2),
        gmap_img_fts=torch.randn(2, 5, width),
        gmap_pos_fts=torch.randn(2, 5, 7),
        gmap_masks=torch.tensor([
            [True, True, True, True, True],
            [True, True, True, True, False],
        ]),
        gmap_visited_masks=torch.tensor([
            [False, True, True, False, False],
            [False, True, False, False, False],
        ]),
        gmap_pair_dists=torch.zeros(2, 5, 5),
        gmap_task_embeddings=torch.ones(2, 5, dtype=torch.long),
        gmap_transport_views=torch.randn(2, 5, 3, width),
        gmap_transport_masks=view_masks,
    )


def exercise_model(model):
    torch.manual_seed(15)
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
    finite = torch.isfinite(expected)
    mismatch = (enabled['global_logits'][finite] - expected[finite]).abs().max()
    if mismatch.item() > 1e-7 or not torch.equal(
            torch.isneginf(enabled['global_logits']), torch.isneginf(expected)):
        raise RuntimeError('Zero-initialized E15 is not E0')
    action_mismatch = (
        enabled['factorized_greedy_actions'] != disabled.argmax(-1)
    ).sum()
    if action_mismatch.item() != 0:
        raise RuntimeError('Zero-initialized E15 changed E0 greedy actions')
    if enabled['transport_residual_norm'].item() != 0:
        raise RuntimeError('E15 residual is not zero initialized')

    objective = model.forward_navigation(**nav_inputs)['global_logits'][:, 3].sum()
    objective.backward()
    if model.factorized_landmark.output.weight.grad.abs().sum() == 0:
        raise RuntimeError('Navigation objective did not reach E15 output')
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.factorized_landmark.output.weight.normal_(std=0.01)
    outputs = model.forward_navigation(**nav_inputs)
    stop_mismatch = (
        outputs['global_logits'].exp()[:, 0] -
        torch.softmax(outputs['base_global_logits'], dim=-1)[:, 0]
    ).abs().max()
    if stop_mismatch.item() > 1e-7:
        raise RuntimeError('E15 changed the frozen E0 STOP probability')
    if not torch.equal(
            outputs['factorized_greedy_actions'] == 0,
            outputs['base_global_logits'].argmax(-1) == 0):
        raise RuntimeError('E15 changed the frozen E0 STOP decision')
    outputs['global_logits'][:, 3].sum().backward()
    for name, parameter in model.named_parameters():
        if (not name.startswith('factorized_landmark.') and
                parameter.grad is not None):
            raise RuntimeError('E15 gradient reached frozen E0: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)
    for name in ('view_projection.weight', 'text_projection.weight'):
        parameter = dict(model.factorized_landmark.named_parameters())[name]
        if parameter.grad.abs().sum() == 0:
            raise RuntimeError('Navigation objective did not reach E15 ' + name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 590848:
        raise RuntimeError('E15 expected exactly 590,848 trainable parameters')
    return trainable, mismatch.item(), action_mismatch.item(), stop_mismatch.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.factorized_landmark_size', '128',
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

    trainable, mismatch, action_mismatch, stop_mismatch = exercise_model(model)
    print('E15_MODEL_SMOKE_PASSED trainable=%d max_distribution_mismatch=%g '
          'action_mismatch=%d stop_probability_mismatch=%g prepool_replay=1 '
          'frozen_gradient=0' % (
              trainable, mismatch, action_mismatch, stop_mismatch
          ))


if __name__ == '__main__':
    main()
