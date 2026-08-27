#!/usr/bin/env python3
import argparse
import os

import torch


EXPECTED_PARAMETERS = 50619


def main():
    parser = argparse.ArgumentParser(
        description='Verify the production ETP-R1 model with Gaussian BEV.'
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
        'MODEL.candidate_scorer_hidden_size', '0',
        'MODEL.gaussian_bev_hidden_size', '32',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    model.eval()
    field = model.gaussian_bev
    if field is None:
        raise RuntimeError('Gaussian BEV was not created')

    trainable_count = sum(parameter.numel() for parameter in field.parameters())
    if trainable_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d Gaussian-BEV parameters, got %d' %
            (EXPECTED_PARAMETERS, trainable_count)
        )
    if torch.count_nonzero(field.output.weight).item() != 0:
        raise RuntimeError('Gaussian-BEV output is not zero-initialized')

    torch.manual_seed(0)
    txt_embeds = torch.randn(1, 4, 768)
    txt_masks = torch.ones(1, 4, dtype=torch.bool)
    gmap_step_ids = torch.zeros(1, 3, dtype=torch.long)
    gmap_img_fts = torch.randn(1, 3, 768)
    gmap_pos_fts = torch.zeros(1, 3, 12)
    gmap_pos_fts[:, :, 1] = 1.0
    gmap_pos_fts[:, :, 3] = 1.0
    gmap_pos_fts[:, 1:, 4] = torch.tensor([0.1, 0.2])
    gmap_pos_fts[:, 1:, 7] = 0.2
    gmap_pos_fts[:, 1:, 9] = 0.4
    gmap_pos_fts[:, 1:, 10] = 0.5
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
        model.gaussian_bev = None
        baseline_logits = model.forward_navigation(
            *navigation_inputs
        )['global_logits']
        model.gaussian_bev = field
    if not torch.equal(enabled_logits, baseline_logits):
        raise RuntimeError('E4 changed the production navigation logits')

    representations = torch.randn(2, 4, 1536, requires_grad=True)
    positions = gmap_pos_fts.expand(2, -1, -1)
    positions = torch.cat([positions, positions[:, -1:]], dim=1)
    masks = torch.ones(2, 4, dtype=torch.bool)
    visited = torch.zeros(2, 4, dtype=torch.bool)
    residual = field(representations, positions, masks, visited)
    if torch.count_nonzero(residual).item() != 0:
        raise RuntimeError('Zero E4 residual changed baseline logits')

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in field.parameters():
        parameter.requires_grad = True
    field(representations, positions, masks, visited).sum().backward()
    if representations.grad is not None:
        raise RuntimeError('Gaussian BEV back-propagated into the backbone')
    if torch.count_nonzero(field.output.weight.grad).item() == 0:
        raise RuntimeError('Gaussian-BEV output has no gradient')

    print(
        'E4_MODEL_SMOKE_PASSED trainable=%d max_initial_residual=0 '
        'backbone_gradient=0' % trainable_count
    )


if __name__ == '__main__':
    main()
