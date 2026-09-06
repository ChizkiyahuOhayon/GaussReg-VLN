#!/usr/bin/env python3
import argparse

import torch


EXPECTED_PARAMETERS = 1857


def main():
    parser = argparse.ArgumentParser(
        description='Check a trained E11 GeoToken-R1 checkpoint.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    tensors = {
        name: tensor for name, tensor in state_dict.items()
        if '.geo_token.' in name
    }
    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d GeoToken parameters, got %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError('Non-finite GeoToken tensor: %s' % name)

    output = {
        name: tensor for name, tensor in tensors.items()
        if name.endswith('geo_token.output.weight')
    }
    if len(output) != 1:
        raise RuntimeError('Expected one GeoToken output weight')
    name, weight = next(iter(output.items()))
    if torch.count_nonzero(weight).item() == 0:
        raise RuntimeError('GeoToken output remained zero')
    print(
        'name=%s shape=%s max_abs=%.8g l2_norm=%.8g parameters=%d' %
        (name, tuple(weight.shape), weight.abs().max().item(),
         weight.norm().item(), parameter_count)
    )


if __name__ == '__main__':
    main()
