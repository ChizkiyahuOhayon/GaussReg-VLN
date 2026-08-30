#!/usr/bin/env python3
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F


EXPECTED_PARAMETERS = 200449


def main():
    parser = argparse.ArgumentParser(
        description='Verify the production ETP-R1 hindsight-stop model.'
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
    from vlnce_baselines.models.etp.ETP_R1_vlnbert_init import get_vlnbert_models

    config = get_config(args.config, [
        'MODEL.gauss_feat_size', '0',
        'MODEL.candidate_scorer_hidden_size', '0',
        'MODEL.gaussian_bev_hidden_size', '0',
        'MODEL.anchor_repair_hidden_size', '0',
        'MODEL.hindsight_stop_hidden_size', '128',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    model.eval()
    head = model.hindsight_stop
    if head is None:
        raise RuntimeError('Hindsight-stop head was not created')

    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d hindsight-stop parameters, got %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )
    if torch.count_nonzero(head.output.weight).item() != 0:
        raise RuntimeError('Hindsight-stop output is not zero-initialized')
    if torch.count_nonzero(head.output.bias).item() != 0:
        raise RuntimeError('Hindsight-stop bias is not zero-initialized')

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
    navigation_inputs = (
        txt_embeds, txt_masks,
        [[None, 'visited-1', 'visited-2', 'candidate', None]] * 2,
        gmap_step_ids, gmap_img_fts, gmap_pos_fts, gmap_masks,
        gmap_visited_masks, gmap_pair_dists, gmap_task_embeddings,
    )
    with torch.no_grad():
        outputs = model.forward_navigation(*navigation_inputs)
        stop_logits = outputs['hindsight_stop_logits']
        base_logits = outputs['base_global_logits']
        model.hindsight_stop = None
        baseline_logits = model.forward_navigation(
            *navigation_inputs
        )['global_logits']
        model.hindsight_stop = head

    if not torch.equal(outputs['global_logits'], baseline_logits):
        raise RuntimeError('E6 changed the frozen E0 navigation logits')
    if not torch.equal(base_logits, baseline_logits):
        raise RuntimeError('E6 changed the stored E0 navigation logits')
    if stop_logits.argmax(dim=-1).tolist() != [0, 0]:
        raise RuntimeError('E6 does not initially select CONTINUE')
    if not torch.isneginf(stop_logits[0, 3:]).all():
        raise RuntimeError('E6 exposed a ghost or padding stop action')

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    model.train()
    logits = model.forward_navigation(*navigation_inputs)[
        'hindsight_stop_logits'
    ]
    F.cross_entropy(logits, torch.tensor([2, 1])).backward()
    backbone_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if 'hindsight_stop.' not in name and parameter.grad is not None
    ]
    if backbone_gradients:
        raise RuntimeError('Hindsight-stop loss reached the frozen backbone')
    if torch.count_nonzero(head.output.weight.grad).item() == 0:
        raise RuntimeError('Hindsight-stop output has no gradient')

    print(
        'E6_MODEL_SMOKE_PASSED trainable=%d initial_stop_mismatch=0 '
        'backbone_gradient=0' % parameter_count
    )


if __name__ == '__main__':
    main()
