#!/usr/bin/env python3
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Instantiate the real ETP-R1 model and verify the E2 residual.'
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
        'MODEL.pretrained_path', args.pretrained,
    ])
    model = get_vlnbert_models(config.MODEL)
    model.eval()

    encoder = model.global_encoder
    gaussian = encoder.gmap_gauss_embedding
    if gaussian is None or tuple(gaussian.weight.shape) != (768, 5):
        raise RuntimeError('Expected a bias-free Linear(5, 768) Gaussian residual')
    if torch.count_nonzero(gaussian.weight).item() != 0:
        raise RuntimeError('Gaussian residual is not zero-initialized')

    torch.manual_seed(0)
    position_features = torch.randn(2, 4, 12)
    with torch.no_grad():
        expected = encoder.gmap_pos_embeddings(position_features[..., :7])
        actual = encoder.position_embedding(position_features)
    max_abs_difference = (actual - expected).abs().max().item()
    if max_abs_difference != 0.0:
        raise RuntimeError(
            'Zero residual changed the position embedding by %.8g' %
            max_abs_difference
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    gaussian.weight.requires_grad = True
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if len(trainable) != 1 or trainable_count != 3840:
        raise RuntimeError(
            'Expected one 3,840-parameter trainable tensor, got %d and %d' %
            (len(trainable), trainable_count)
        )

    encoder.position_embedding(position_features).square().mean().backward()
    gradient_nonzero = torch.count_nonzero(gaussian.weight.grad).item()
    if gradient_nonzero == 0:
        raise RuntimeError('Gaussian residual has no gradient')

    print(
        'E2_MODEL_SMOKE_PASSED '
        'shape=(768, 5) trainable=3840 max_abs_difference=0 '
        'gradient_nonzero=%d' % gradient_nonzero
    )


if __name__ == '__main__':
    main()
