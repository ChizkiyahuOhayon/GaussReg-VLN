#!/usr/bin/env python3
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Report the learned E3 candidate-scorer output layer.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu')
    state = payload.get('state_dict', payload)
    tensors = {
        name: value for name, value in state.items()
        if '.candidate_scorer.output.' in name
    }
    if len(tensors) != 2:
        raise RuntimeError(
            'Expected candidate-scorer output weight and bias, found %s' %
            sorted(tensors)
        )
    max_abs = 0.0
    for name, value in sorted(tensors.items()):
        if not torch.isfinite(value).all():
            raise RuntimeError('Non-finite candidate-scorer tensor: %s' % name)
        tensor_max_abs = value.abs().max().item()
        max_abs = max(max_abs, tensor_max_abs)
        print(
            'name=%s shape=%s max_abs=%.8g l2_norm=%.8g' %
            (name, tuple(value.shape), tensor_max_abs, value.norm().item())
        )
    if max_abs == 0.0:
        raise RuntimeError('Candidate-scorer output is still exactly zero')


if __name__ == '__main__':
    main()
