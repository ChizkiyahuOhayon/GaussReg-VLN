#!/usr/bin/env python3
import argparse

import torch


WEIGHT_SUFFIX = 'gmap_gauss_embedding.weight'
EXPECTED_SHAPE = (768, 5)


def main():
    parser = argparse.ArgumentParser(
        description='Verify that an E2 checkpoint contains an updated Gaussian residual.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu')
    state_dict = payload.get('state_dict', payload)
    matches = [
        (name, value) for name, value in state_dict.items()
        if name.endswith(WEIGHT_SUFFIX)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            'Expected one %s tensor, found %d' % (WEIGHT_SUFFIX, len(matches))
        )

    name, weight = matches[0]
    if tuple(weight.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            'Expected Gaussian weight shape %s, got %s' %
            (EXPECTED_SHAPE, tuple(weight.shape))
        )

    max_abs = weight.abs().max().item()
    l2_norm = weight.norm().item()
    print('name=%s shape=%s max_abs=%.8g l2_norm=%.8g' % (
        name, tuple(weight.shape), max_abs, l2_norm
    ))
    if max_abs == 0.0:
        raise RuntimeError('Gaussian residual is still exactly zero')


if __name__ == '__main__':
    main()
