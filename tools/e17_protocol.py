#!/usr/bin/env python3
"""E17 provenance, checkpoint contract and full-evaluation gate."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import e15_protocol as base
from tools import e16_protocol as gate

evaluate_results = gate.evaluate_results


def preflight(args):
    import torch
    from vlnce_baselines.config.default import get_config

    train_path, train = base.dataset_episodes('train', '_10')
    eval_path, evaluation = base.dataset_episodes('val_unseen')
    eval_ids = {str(episode['episode_id']) for episode in evaluation}
    if len(evaluation) != 1839 or len(eval_ids) != 1839:
        raise ValueError('Expected the full 1,839-episode R2R val_unseen dataset')
    if not train:
        raise ValueError('Empty train subset')
    for episode in train + evaluation:
        scene = episode['scene_id'].replace('data/scene_datasets/', '', 1)
        if not (Path('data/scene_datasets') / scene).is_file():
            raise FileNotFoundError('Missing scene: ' + scene)
    assets = [
        args.baseline, args.pretrained, train_path, eval_path,
        base.DATASET / 'val_unseen/val_unseen_gt.json.gz',
        'data/wp_pred/check_cwp_bestdist_hfov90',
        'data/ddppo-models/gibson-2plus-resnet50.pth',
        'run_r2r/iter_train.yaml', 'run_r2r/r2r_vlnce.yaml',
        'run_r2r/e17_closed_loop_1gpu.bash',
        'tools/e17_protocol.py', 'tools/smoke_e17_model.py',
        'vlnce_baselines/monotonic_factorized_landmark.py',
        'vlnce_baselines/factorized_landmark.py',
        'docs/e17_monotonic_factorized_landmark.md',
    ]
    if subprocess.check_output(
            ['git', 'diff', 'HEAD', '--name-only'],
            universal_newlines=True).strip():
        raise RuntimeError('Commit tracked code changes before a reproducible E17 run')
    report = {
        'experiment': 'E17-MFLR',
        'utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], universal_newlines=True
        ).strip(),
        'python': sys.version, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'train_episodes': len(train), 'eval_episodes': len(evaluation),
        'seed': get_config('run_r2r/iter_train.yaml').TASK_CONFIG.SEED,
        'asset_sha256': {str(path): base.sha256(path) for path in assets},
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('E17_PREFLIGHT_OK ' + json.dumps(report, sort_keys=True))


def check_checkpoint(args):
    import torch

    spec = importlib.util.spec_from_file_location(
        'landmark_transport_check',
        ROOT / 'vlnce_baselines/landmark_transport.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    baseline = torch.load(args.baseline, map_location='cpu')
    state = {key.replace('net.module.', 'net.'): value
             for key, value in checkpoint['state_dict'].items()}
    frozen = {key.replace('net.module.', 'net.'): value
              for key, value in baseline['state_dict'].items()}
    prefix = 'net.vln_bert.factorized_landmark.'
    transport = {key[len(prefix):]: value for key, value in state.items()
                 if key.startswith(prefix)}
    learned = {key: value for key, value in state.items()
               if not key.startswith(prefix)}
    if set(learned) != set(frozen) or any(
            not torch.equal(learned[key], frozen[key]) for key in frozen):
        raise RuntimeError('E17 changed frozen E0 tensors or used a wrong baseline')
    expected = module.LandmarkTransport(768, 128).state_dict()
    if set(transport) != set(expected) or any(
            transport[key].shape != expected[key].shape for key in expected):
        raise RuntimeError('E17 transport is missing, partial or misconfigured')
    if sum(tensor.numel() for tensor in transport.values()) != 590848:
        raise RuntimeError('E17 checkpoint has the wrong transport parameter count')
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise RuntimeError('E17 checkpoint contains nonfinite tensors')
    if (checkpoint.get('iteration') != args.iteration or
            checkpoint.get('e17_optimizer_updates') != args.iteration):
        raise RuntimeError('E17 iteration or optimizer-update count is wrong')
    if (torch.count_nonzero(transport['output.weight']).item() == 0 and
            torch.count_nonzero(transport['output.bias']).item() == 0):
        raise RuntimeError('E17 output stayed at its zero initialization')

    config = checkpoint['config']
    excluded_modes = [
        'gauss_only', 'candidate_scorer_only', 'gaussian_bev_only',
        'anchor_repair_only', 'hindsight_stop_only', 'terminal_commit_only',
        'success_set_commit', 'setwise_group_policy', 'frontier_advantage',
        'geo_token_only', 'successor_only', 'instruction_coverage_only',
        'landmark_transport_only', 'factorized_landmark_only',
        'budgeted_factorized_landmark_only',
    ]
    excluded_sizes = [
        'gauss_feat_size', 'candidate_scorer_hidden_size',
        'gaussian_bev_hidden_size', 'anchor_repair_hidden_size',
        'hindsight_stop_hidden_size', 'terminal_commit_hidden_size',
        'geo_token_hidden_size', 'successor_hidden_size',
        'instruction_coverage_hidden_size', 'landmark_transport_size',
    ]
    if (not config.GRPO.monotonic_factorized_landmark_only or
            any(getattr(config.GRPO, name, False) for name in excluded_modes) or
            any(getattr(config.MODEL, name, 0) != 0
                for name in excluded_sizes) or
            config.MODEL.factorized_landmark_size != 128 or
            config.MODEL.factorized_landmark_budgeted or
            not config.MODEL.factorized_landmark_monotonic or
            config.GRPO.sample_num != 8 or config.GRPO.update_epochs != 1 or
            abs(config.GRPO.lr - 1e-4) > 1e-12 or
            abs(config.GRPO.grpo_beta - 0.04) > 1e-12 or
            config.GRPO.back_algo != 'control' or config.GRPO.is_requeue or
            config.GRPO.enable_amp or config.GRPO.enable_all_dropouts or
            config.GRPO.dropout_in_sampling or config.GRPO.waypoint_aug or
            config.TASK_CONFIG.DATASET.SPLIT != 'train' or
            config.TASK_CONFIG.DATASET.SUFFIX != '_10'):
        raise RuntimeError('Checkpoint training configuration differs from E17 protocol')
    print('E17_CHECKPOINT_OK iteration=%d updates=%d parameters=590848 '
          'frozen_E0_equal=1 sha256=%s' % (
              args.iteration, checkpoint['e17_optimizer_updates'],
              base.sha256(args.checkpoint)
          ))


def results(args):
    dataset_path, episodes = base.dataset_episodes('val_unseen')
    expected = {str(episode['episode_id']) for episode in episodes}
    if len(expected) != 1839:
        raise ValueError('Full evaluation requires 1,839 unique episodes')
    directory = Path(args.directory)
    files = list(directory.glob('stats_ep_ckpt_*_val_unseen_r0_w1.json'))
    aggregates = list(directory.glob('stats_ckpt_*_val_unseen.json'))
    if len(files) != 1 or len(aggregates) != 1:
        raise ValueError('Expected exactly one fresh full-evaluation result pair')
    rows = json.loads(files[0].read_text())
    report = gate.evaluate_results(rows, expected)
    aggregate = json.loads(aggregates[0].read_text())
    for display, key in base.METRICS.items():
        scale = 1 if display == 'NE' else 100
        if (not math.isfinite(aggregate[key]) or
                abs(report['metrics'][display] / scale - aggregate[key]) > 1e-8):
            raise ValueError('Aggregate and per-episode metrics disagree: ' + key)
    report.update({
        'checkpoint': args.checkpoint,
        'checkpoint_sha256': base.sha256(args.checkpoint),
        'per_episode_file': str(files[0]),
        'per_episode_sha256': base.sha256(files[0]),
        'dataset_sha256': base.sha256(dataset_path),
        'experiment': 'E17-MFLR',
        'reference': 'E15-FLR',
        'utc': datetime.now(timezone.utc).isoformat(),
    })
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('E17_FULL_EVAL ' + json.dumps(report, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    pre = subparsers.add_parser('preflight')
    pre.add_argument('--baseline', required=True)
    pre.add_argument('--pretrained', required=True)
    pre.add_argument('--output', required=True)
    check = subparsers.add_parser('checkpoint')
    check.add_argument('checkpoint')
    check.add_argument('--baseline', required=True)
    check.add_argument('--iteration', type=int, required=True)
    result = subparsers.add_parser('results')
    result.add_argument('directory')
    result.add_argument('--checkpoint', required=True)
    result.add_argument('--output', required=True)
    args = parser.parse_args()
    handlers = {'preflight': preflight, 'checkpoint': check_checkpoint,
                'results': results}
    handlers[args.command](args)


if __name__ == '__main__':
    main()
