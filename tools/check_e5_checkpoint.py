#!/usr/bin/env python3
import argparse

import torch


EXPECTED_PARAMETERS = 154972


def main():
    parser = argparse.ArgumentParser(
        description='Check a trained E5 anchor-repair checkpoint.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    tensors = {
        name: value for name, value in state_dict.items()
        if '.anchor_repair.' in name
    }
    parameter_count = sum(value.numel() for value in tensors.values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d anchor-repair parameters, found %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )

    output_weights = {
        name: value for name, value in tensors.items()
        if name.endswith('anchor_repair.output.weight')
    }
    if len(output_weights) != 1:
        raise RuntimeError('Expected one anchor-repair output weight')
    name, weight = next(iter(output_weights.items()))
    if not torch.isfinite(weight).all():
        raise RuntimeError('Anchor-repair output contains non-finite values')
    if torch.count_nonzero(weight).item() == 0:
        raise RuntimeError('Anchor-repair output remained zero')

    print(
        'name=%s shape=%s max_abs=%.8g l2_norm=%.8g parameters=%d' %
        (name, tuple(weight.shape), weight.abs().max().item(),
         weight.norm().item(), parameter_count)
    )


if __name__ == '__main__':
    main()
