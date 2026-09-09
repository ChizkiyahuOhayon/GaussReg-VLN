#!/usr/bin/env python3
"""Exercise E14 through the production model after a strict E0 load."""

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
    torch.manual_seed(14)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.landmark_transport.parameters():
        parameter.requires_grad_(True)
    nav_inputs = inputs(model.config.hidden_size)

    with torch.no_grad():
        enabled = model.forward_navigation(**nav_inputs)
        transport = model.landmark_transport
        model.landmark_transport = None
        disabled = model.forward_navigation(**{
            key: value for key, value in nav_inputs.items()
            if not key.startswith('gmap_transport_')
        })['global_logits']
        model.landmark_transport = transport
    if not torch.equal(enabled['global_logits'], disabled):
        raise RuntimeError('Zero-initialized E14 is not exactly E0')

    finite = torch.isfinite(disabled)
    mismatch = (enabled['global_logits'][finite] - disabled[finite]).abs().max()
    action_mismatch = (
        enabled['global_logits'].argmax(-1) != disabled.argmax(-1)
    ).sum()
    if enabled['transport_residual_norm'].item() != 0:
        raise RuntimeError('E14 residual is not zero initialized')

    objective = torch.nn.functional.log_softmax(
        model.forward_navigation(**nav_inputs)['global_logits'], dim=1
    )[:, 3].sum()
    objective.backward()
    if model.landmark_transport.output.weight.grad.abs().sum() == 0:
        raise RuntimeError('Navigation objective did not reach E14 output')
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.landmark_transport.output.weight.normal_(std=0.01)
    model.forward_navigation(**nav_inputs)['global_logits'][:, 3].sum().backward()
    for name, parameter in model.named_parameters():
        if not name.startswith('landmark_transport.') and parameter.grad is not None:
            raise RuntimeError('E14 gradient reached frozen E0: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)
    for name in ('view_projection.weight', 'text_projection.weight'):
        if dict(model.landmark_transport.named_parameters())[name].grad.abs().sum() == 0:
            raise RuntimeError('Navigation objective did not reach E14 ' + name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 590848:
        raise RuntimeError('E14 expected exactly 590,848 trainable parameters')
    return trainable, mismatch.item(), action_mismatch.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.landmark_transport_size', '128',
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
        'landmark_transport.' + key
        for key in model.landmark_transport.state_dict()
    }
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError('Strict E0 model load mismatch: ' + str(missing))
    model.landmark_transport.reset_output()
    del checkpoint, weights

    trainable, mismatch, action_mismatch = exercise_model(model)
    print('E14_MODEL_SMOKE_PASSED trainable=%d max_logit_mismatch=%g '
          'action_mismatch=%d prepool_replay=1 frozen_gradient=0' % (
              trainable, mismatch, action_mismatch
          ))


if __name__ == '__main__':
    main()
