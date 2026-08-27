#!/usr/bin/env python3
import argparse

import torch


EXPECTED_PARAMETERS = 50619


def main():
    parser = argparse.ArgumentParser(
        description='Report the learned E4 Gaussian-BEV parameters.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu')
    state = payload.get('state_dict', payload)
    tensors = {
        name: value for name, value in state.items()
        if '.gaussian_bev.' in name
    }
    parameter_count = sum(value.numel() for value in tensors.values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d Gaussian-BEV parameters, found %d in %s' %
            (EXPECTED_PARAMETERS, parameter_count, sorted(tensors))
        )

    output = {
        name: value for name, value in tensors.items()
        if name.endswith('gaussian_bev.output.weight')
    }
    if len(output) != 1:
        raise RuntimeError(
            'Expected one Gaussian-BEV output tensor, found %s' %
            sorted(output)
        )

    for name, value in sorted(tensors.items()):
        if not torch.isfinite(value).all():
            raise RuntimeError('Non-finite Gaussian-BEV tensor: %s' % name)
    name, value = next(iter(output.items()))
    max_abs = value.abs().max().item()
    print(
        'name=%s shape=%s max_abs=%.8g l2_norm=%.8g parameters=%d' %
        (name, tuple(value.shape), max_abs, value.norm().item(),
         parameter_count)
    )
    if max_abs == 0.0:
        raise RuntimeError('Gaussian-BEV output is still exactly zero')


if __name__ == '__main__':
    main()
