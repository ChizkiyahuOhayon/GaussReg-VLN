#!/usr/bin/env python3
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch


EXPECTED_PARAMETERS = 1857


def main():
    parser = argparse.ArgumentParser(
        description='Verify the production E11 GeoToken-R1 model.'
    )
    parser.add_argument('--config', default='run_r2r/iter_train.yaml')
    parser.add_argument(
        '--pretrained',
        default=(
            'pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/'
            'store2/model_step_367500.pt'
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pretrained):
        raise FileNotFoundError('Missing pretrained checkpoint: %s' % args.pretrained)

    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.geo_token import gaussian_free_space_tokens
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config(args.config, [
        'MODEL.gauss_feat_size', '0',
        'MODEL.candidate_scorer_hidden_size', '0',
        'MODEL.gaussian_bev_hidden_size', '0',
        'MODEL.anchor_repair_hidden_size', '0',
        'MODEL.hindsight_stop_hidden_size', '0',
        'MODEL.terminal_commit_hidden_size', '0',
        'MODEL.geo_token_hidden_size', '64',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    model.eval()
    geo_token = model.geo_token
    if geo_token is None:
        raise RuntimeError('GeoToken residual was not created')

    parameter_count = sum(
        parameter.numel() for parameter in geo_token.parameters()
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d GeoToken parameters, got %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )
    if torch.count_nonzero(geo_token.output.weight).item() != 0:
        raise RuntimeError('GeoToken output is not zero-initialized')

    depth = torch.full((2, 32, 32), 0.5)
    candidate_tokens, candidate_masks = gaussian_free_space_tokens(
        depth,
        heading_offsets=torch.tensor([0.0, 0.25]),
        waypoint_distances=torch.tensor([2.0, 3.0]),
    )
    if candidate_tokens.shape != (2, 3, 8):
        raise RuntimeError('Unexpected candidate GeoToken shape')
    if not torch.isfinite(candidate_tokens).all() or not candidate_masks.any():
        raise RuntimeError('Candidate GeoTokens are invalid')

    torch.manual_seed(0)
    txt_embeds = torch.randn(2, 4, 768)
    txt_masks = torch.ones(2, 4, dtype=torch.bool)
    gmap_step_ids = torch.tensor([
        [0, 1, 2, 0, 0],
        [0, 1, 2, 3, 0],
    ])
    gmap_img_fts = torch.randn(2, 5, 768)
    gmap_pos_fts = torch.zeros(2, 5, 7)
    gmap_masks = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, True, True],
    ])
    gmap_visited_masks = torch.tensor([
        [False, True, True, False, False],
        [False, True, True, True, False],
    ])
    gmap_pair_dists = torch.zeros(2, 5, 5)
    gmap_task_embeddings = torch.zeros(2, 5, dtype=torch.long)
    gmap_geo_tokens = torch.randn(2, 5, 3, 8)
    gmap_geo_masks = torch.ones(2, 5, 3, dtype=torch.bool)
    navigation_inputs = (
        txt_embeds, txt_masks,
        [[None, 'visited-1', 'visited-2', 'candidate', None]] * 2,
        gmap_step_ids, gmap_img_fts, gmap_pos_fts, gmap_masks,
        gmap_visited_masks, gmap_pair_dists, gmap_task_embeddings,
    )
    with torch.no_grad():
        outputs = model.forward_navigation(
            *navigation_inputs,
            gmap_geo_tokens=gmap_geo_tokens,
            gmap_geo_masks=gmap_geo_masks,
        )
        model.geo_token = None
        baseline_logits = model.forward_navigation(
            *navigation_inputs
        )['global_logits']
        model.geo_token = geo_token

    if not torch.equal(outputs['global_logits'], baseline_logits):
        raise RuntimeError('E11 changed the initial E0 navigation logits')

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in geo_token.parameters():
        parameter.requires_grad = True
    model.train()
    train_tokens = gmap_geo_tokens.clone().requires_grad_(True)
    logits = model.forward_navigation(
        *navigation_inputs,
        gmap_geo_tokens=train_tokens,
        gmap_geo_masks=gmap_geo_masks,
    )['global_logits']
    finite_logits = logits[torch.isfinite(logits)]
    finite_logits.sum().backward()
    backbone_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if 'geo_token.' not in name and parameter.grad is not None
    ]
    if backbone_gradients:
        raise RuntimeError('GeoToken loss reached the frozen backbone')
    if train_tokens.grad is not None:
        raise RuntimeError('GeoToken residual did not detach geometry inputs')
    if torch.count_nonzero(geo_token.output.weight.grad).item() == 0:
        raise RuntimeError('GeoToken output has no gradient')

    print(
        'E11_MODEL_SMOKE_PASSED trainable=%d initial_action_mismatch=0 '
        'backbone_gradient=0 token_shape=%s' %
        (parameter_count, tuple(candidate_tokens.shape))
    )


if __name__ == '__main__':
    main()
