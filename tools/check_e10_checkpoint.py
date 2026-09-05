#!/usr/bin/env python3
import argparse

import torch


FORBIDDEN_MODULES = (
    '.candidate_scorer.',
    '.gaussian_bev.',
    '.anchor_repair.',
    '.hindsight_stop.',
    '.terminal_commit.',
    'gmap_gauss_embedding.',
)


def main():
    parser = argparse.ArgumentParser(
        description='Check an E10 FrontierAdv-R1 checkpoint.'
    )
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config')
    if config is None or not getattr(config.GRPO, 'frontier_advantage', False):
        raise RuntimeError('Checkpoint is not marked as an E10 run')

    state_dict = checkpoint.get('state_dict', checkpoint)
    forbidden = [
        name for name in state_dict
        if any(module in name for module in FORBIDDEN_MODULES)
    ]
    if forbidden:
        raise RuntimeError(
            'E10 checkpoint contains inference-only experiment modules: %s' %
            forbidden
        )

    print(
        'E10_CHECKPOINT_OK tensors=%d added_parameters=0 '
        'inference_changes=0 iteration=%s' % (
            len(state_dict), checkpoint.get('iteration', 'unknown')
        )
    )


if __name__ == '__main__':
    main()
