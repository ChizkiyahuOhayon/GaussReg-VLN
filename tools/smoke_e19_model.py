#!/usr/bin/env python3
"""Exercise E19 through the production model after a strict E0 load."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def navigation_inputs(width):
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
        gmap_transport_masks=torch.tensor([
            [[False] * 3, [True] * 3, [True, False, False],
             [True, True, False], [True, False, False]],
            [[False] * 3, [True, True, False], [True, False, False],
             [True, False, False], [False] * 3],
        ]),
        gmap_local_geometry=torch.randn(2, 5, 128, 3),
        gmap_local_geometry_masks=torch.tensor([
            [False, False, False, True, True],
            [False, False, False, True, False],
        ]),
    )


def exercise_model(model):
    torch.manual_seed(19)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    branches = (model.factorized_landmark, model.transient_geometry)
    for branch in branches:
        for parameter in branch.parameters():
            parameter.requires_grad_(True)
    inputs = navigation_inputs(model.config.hidden_size)

    with torch.no_grad():
        enabled = model.forward_navigation(**inputs)
        model.factorized_landmark = None
        model.transient_geometry = None
        disabled = model.forward_navigation(**{
            key: value for key, value in inputs.items()
            if not (key.startswith('gmap_transport_') or
                    key.startswith('gmap_local_geometry'))
        })['global_logits']
        model.factorized_landmark, model.transient_geometry = branches
    expected = torch.log_softmax(disabled, dim=-1)
    finite = torch.isfinite(expected)
    mismatch = (enabled['global_logits'][finite] - expected[finite]).abs().max()
    if mismatch.item() > 1e-7 or not torch.equal(
            torch.isneginf(enabled['global_logits']), torch.isneginf(expected)):
        raise RuntimeError('Zero-initialized E19 is not E0')
    if enabled['geometry_residual_norm'].item() != 0:
        raise RuntimeError('E19 geometry residual is not zero initialized')

    with torch.no_grad():
        for branch in branches:
            branch.output.weight.normal_(std=0.01)
    outputs = model.forward_navigation(**inputs)
    stop_mismatch = (
        outputs['global_logits'].exp()[:, 0] -
        torch.softmax(outputs['base_global_logits'], dim=-1)[:, 0]
    ).abs().max()
    if stop_mismatch.item() > 1e-7:
        raise RuntimeError('E19 changed the frozen E0 STOP probability')
    if not torch.equal(
            outputs['factorized_greedy_actions'] == 0,
            outputs['base_global_logits'].argmax(-1) == 0):
        raise RuntimeError('E19 changed the frozen E0 STOP decision')
    outputs['global_logits'][:, 3].sum().backward()
    for name, parameter in model.named_parameters():
        allowed = (name.startswith('factorized_landmark.') or
                   name.startswith('transient_geometry.'))
        if not allowed and parameter.grad is not None:
            raise RuntimeError('E19 gradient reached frozen E0: ' + name)
        if allowed and parameter.grad is None:
            raise RuntimeError('Navigation objective missed E19: ' + name)
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Nonfinite gradient: ' + name)

    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    if trainable != 706944:
        raise RuntimeError('E19 expected exactly 706,944 trainable parameters')
    return trainable, mismatch.item(), stop_mismatch.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config('run_r2r/iter_train.yaml', [
        'MODEL.factorized_landmark_size', '128',
        'MODEL.transient_geometry_size', '128',
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
    expected.update({
        'transient_geometry.' + key
        for key in model.transient_geometry.state_dict()
    })
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise RuntimeError('Strict E0 model load mismatch: ' + str(missing))
    model.factorized_landmark.reset_output()
    model.transient_geometry.reset_output()
    del checkpoint, weights

    trainable, mismatch, stop_mismatch = exercise_model(model)
    print('E19_MODEL_SMOKE_PASSED trainable=%d '
          'max_distribution_mismatch=%g stop_probability_mismatch=%g '
          'current_geometry_only=1 frozen_gradient=0' % (
              trainable, mismatch, stop_mismatch
          ))


if __name__ == '__main__':
    main()
