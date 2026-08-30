#!/usr/bin/env python3
import argparse
import math

import torch


EXPECTED_PARAMETERS = 200449


def main():
    parser = argparse.ArgumentParser(
        description='Verify an E6 hindsight-stop checkpoint.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    tensors = {
        name: tensor for name, tensor in state_dict.items()
        if '.hindsight_stop.' in name
    }
    if not tensors:
        raise RuntimeError('Checkpoint has no hindsight-stop parameters')
    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            'Expected %d hindsight-stop parameters, got %d' %
            (EXPECTED_PARAMETERS, parameter_count)
        )
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
        raise RuntimeError('Checkpoint contains a non-finite hindsight-stop value')

    output = next(
        (tensor for name, tensor in tensors.items()
         if name.endswith('hindsight_stop.output.weight')),
        None,
    )
    if output is None:
        raise RuntimeError('Checkpoint is missing hindsight_stop.output.weight')
    max_abs = output.abs().max().item()
    if max_abs == 0.0:
        raise RuntimeError('Hindsight-stop output remained zero')
    l2_norm = math.sqrt(output.float().square().sum().item())
    print(
        'name=hindsight_stop.output.weight shape=%s max_abs=%.8g '
        'l2_norm=%.8g parameters=%d' %
        (tuple(output.shape), max_abs, l2_norm, parameter_count)
    )


if __name__ == '__main__':
    main()
