#!/usr/bin/env python3
import argparse
import os

import torch


EXPECTED_PARAMETERS = 154972


def main():
    parser = argparse.ArgumentParser(
        description='Verify the production ETP-R1 anchor-repair model.'
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
        'MODEL.gaussian_bev_hidden_size', '0',
        'MODEL.anchor_repair_hidden_size', '64',
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    model.eval()
    repair = model.anchor_repair
    if repair is None:
        raise RuntimeError('Anchor repair was not created')

    parameter_count = sum(parameter.numel() for parameter in repair.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d anchor-repair parameters, got %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )
    if torch.count_nonzero(repair.output.weight).item() != 0:
        raise RuntimeError('Anchor-repair output is not zero-initialized')
    if torch.count_nonzero(repair.output.bias).item() != 0:
        raise RuntimeError('Anchor-repair bias is not zero-initialized')

    torch.manual_seed(0)
    txt_embeds = torch.randn(2, 4, 768)
    txt_masks = torch.ones(2, 4, dtype=torch.bool)
    gmap_step_ids = torch.zeros(2, 4, dtype=torch.long)
    gmap_img_fts = torch.randn(2, 4, 768)
    gmap_pos_fts = torch.zeros(2, 4, 12)
    gmap_pos_fts[:, :, 1] = 1.0
    gmap_pos_fts[:, :, 3] = 1.0
    gmap_pos_fts[:, 1:, 4] = torch.tensor([0.1, 0.2, 0.3])
    gmap_pos_fts[:, 1:, 7] = 0.2
    gmap_pos_fts[:, 1:, 9] = 0.4
    gmap_pos_fts[:, 1:, 10] = 0.5
    gmap_masks = torch.ones(2, 4, dtype=torch.bool)
    gmap_masks[1, 3] = False
    gmap_visited_masks = torch.tensor([
        [False, True, False, False],
        [False, False, True, False],
    ])
    gmap_pair_dists = torch.zeros(2, 4, 4)
    gmap_task_embeddings = torch.zeros(2, 4, dtype=torch.long)
    navigation_inputs = (
        txt_embeds, txt_masks,
        [[None, 'visited', 'candidate-1', 'candidate-2']] * 2,
        gmap_step_ids, gmap_img_fts, gmap_pos_fts, gmap_masks,
        gmap_visited_masks, gmap_pair_dists, gmap_task_embeddings,
    )
    with torch.no_grad():
        outputs = model.forward_navigation(*navigation_inputs)
        repair_logits = outputs['global_logits']
        base_logits = outputs['base_global_logits']
        model.anchor_repair = None
        baseline_logits = model.forward_navigation(
            *navigation_inputs
        )['global_logits']
        model.anchor_repair = repair

    if not torch.equal(base_logits, baseline_logits):
        raise RuntimeError('E5 changed the frozen E0 logits')
    if not torch.equal(
            repair_logits.argmax(dim=-1), base_logits.argmax(dim=-1)):
        raise RuntimeError('E5 changed the initial greedy E0 actions')

    representations = torch.randn(2, 4, 1536, requires_grad=True)
    raw_base_logits = torch.randn(2, 4)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in repair.parameters():
        parameter.requires_grad = True
    logits = repair(
        raw_base_logits, representations, gmap_pos_fts,
        gmap_masks, gmap_visited_masks,
    )
    logits[torch.isfinite(logits)].sum().backward()
    if representations.grad is not None:
        raise RuntimeError('Anchor repair back-propagated into the backbone')
    if torch.count_nonzero(repair.output.weight.grad).item() == 0:
        raise RuntimeError('Anchor-repair output has no gradient')

    print(
        'E5_MODEL_SMOKE_PASSED trainable=%d initial_action_mismatch=0 '
        'backbone_gradient=0' % parameter_count
    )


if __name__ == '__main__':
    main()
