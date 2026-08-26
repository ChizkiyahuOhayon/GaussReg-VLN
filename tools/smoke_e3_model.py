#!/usr/bin/env python3
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Verify the real ETP-R1 model with the E3 candidate scorer.'
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
        'MODEL.gauss_feat_size', '5',
        'MODEL.gauss_residual_scale', '0.0',
        'MODEL.candidate_scorer_hidden_size', '256',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    model.eval()
    scorer = model.candidate_scorer
    if scorer is None:
        raise RuntimeError('Candidate scorer was not created')

    trainable_count = sum(parameter.numel() for parameter in scorer.parameters())
    if trainable_count >= 500000:
        raise RuntimeError('Candidate scorer is not lightweight: %d' % trainable_count)
    if torch.count_nonzero(scorer.output.weight).item() != 0:
        raise RuntimeError('Candidate scorer output is not zero-initialized')
    if torch.count_nonzero(scorer.output.bias).item() != 0:
        raise RuntimeError('Candidate scorer bias is not zero-initialized')

    torch.manual_seed(0)
    txt_embeds = torch.randn(1, 4, 768)
    txt_masks = torch.ones(1, 4, dtype=torch.bool)
    gmap_step_ids = torch.zeros(1, 3, dtype=torch.long)
    gmap_img_fts = torch.randn(1, 3, 768)
    gmap_pos_fts = torch.randn(1, 3, 12)
    gmap_masks = torch.ones(1, 3, dtype=torch.bool)
    gmap_visited_masks = torch.tensor([[False, True, False]])
    gmap_pair_dists = torch.zeros(1, 3, 3)
    gmap_task_embeddings = torch.zeros(1, 3, dtype=torch.long)
    navigation_inputs = (
        txt_embeds, txt_masks, [[None, 'visited', 'candidate']],
        gmap_step_ids, gmap_img_fts, gmap_pos_fts, gmap_masks,
        gmap_visited_masks, gmap_pair_dists, gmap_task_embeddings,
    )
    with torch.no_grad():
        enabled_logits = model.forward_navigation(
            *navigation_inputs
        )['global_logits']
        model.candidate_scorer = None
        baseline_logits = model.forward_navigation(
            *navigation_inputs
        )['global_logits']
        model.candidate_scorer = scorer
    if not torch.equal(enabled_logits, baseline_logits):
        raise RuntimeError('E3 changed the production navigation logits')

    representations = torch.randn(2, 4, 1536, requires_grad=True)
    positions = torch.randn(2, 4, 12)
    visited = torch.tensor([
        [False, True, False, False],
        [False, False, True, False],
    ])
    base_logits = torch.randn(2, 4)
    residual = scorer(representations, positions, visited)
    if not torch.equal(base_logits + residual, base_logits):
        raise RuntimeError('Zero E3 residual changed baseline logits')

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in scorer.parameters():
        parameter.requires_grad = True
    scorer(representations, positions, visited).sum().backward()
    if representations.grad is not None:
        raise RuntimeError('Candidate scorer back-propagated into the backbone')
    if torch.count_nonzero(scorer.output.weight.grad).item() == 0:
        raise RuntimeError('Candidate scorer output has no gradient')

    print(
        'E3_MODEL_SMOKE_PASSED trainable=%d max_initial_residual=0 '
        'backbone_gradient=0' % trainable_count
    )


if __name__ == '__main__':
    main()
