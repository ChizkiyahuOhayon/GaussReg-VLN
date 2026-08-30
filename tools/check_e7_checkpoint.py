#!/usr/bin/env python3
import argparse

import torch


EXPECTED_PARAMETERS = 200321


def main():
    parser = argparse.ArgumentParser(
        description='Check a trained E7 terminal-commit checkpoint.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    tensors = {
        name: tensor for name, tensor in state_dict.items()
        if '.terminal_commit.' in name
    }
    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d terminal-commit parameters, got %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError('Non-finite terminal-commit tensor: %s' % name)

    output = {
        name: tensor for name, tensor in tensors.items()
        if name.endswith('terminal_commit.output.weight')
    }
    if len(output) != 1:
        raise RuntimeError('Expected one terminal-commit output weight')
    name, weight = next(iter(output.items()))
    if torch.count_nonzero(weight).item() == 0:
        raise RuntimeError('Terminal-commit output remained zero')
    print(
        'name=%s shape=%s max_abs=%.8g l2_norm=%.8g parameters=%d' %
        (name, tuple(weight.shape), weight.abs().max().item(),
         weight.norm().item(), parameter_count)
    )


if __name__ == '__main__':
    main()
