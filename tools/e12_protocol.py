#!/usr/bin/env python3
"""E12 asset provenance, frozen-checkpoint contract and full-evaluation gate."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = Path('data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr')
THRESHOLDS = {'NE': 3.93, 'OSR': 71.45, 'SR': 65.31, 'SPL': 56.85,
              'nDTW': 65.8165, 'SDTW': 53.9994}
METRICS = {'NE': 'distance_to_goal', 'OSR': 'oracle_success', 'SR': 'success',
           'SPL': 'spl', 'nDTW': 'ndtw', 'SDTW': 'sdtw'}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_episodes(split, suffix=''):
    path = DATASET / split / (split + suffix + '.json.gz')
    with gzip.open(path, 'rt') as stream:
        episodes = json.load(stream)['episodes']
    return path, episodes


def evaluate_results(rows, expected_ids):
    if set(rows) != expected_ids or not rows:
        raise ValueError('Evaluated episode identities differ from the full dataset')
    metrics = {}
    for display, key in METRICS.items():
        values = [row[key] for row in rows.values()]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               or (display != 'NE' and v > 1) for v in values):
            raise ValueError('Invalid metric or fraction scale: ' + key)
        metrics[display] = sum(values) / len(values) * (1 if display == 'NE' else 100)
    passes = {key: (metrics[key] < bound if key == 'NE' else metrics[key] > bound)
              for key, bound in THRESHOLDS.items()}
    return {'episodes': len(rows), 'metrics': metrics, 'thresholds': THRESHOLDS,
            'passes': passes, 'decision': 'go' if all(passes.values()) else 'no-go'}


def preflight(args):
    import torch
    from vlnce_baselines.config.default import get_config

    train_path, train = dataset_episodes('train', '_10')
    eval_path, evaluation = dataset_episodes('val_unseen')
    if len(evaluation) != 1839 or len({str(e['episode_id']) for e in evaluation}) != 1839:
        raise ValueError('Expected the full 1,839-episode R2R val_unseen dataset')
    if not train:
        raise ValueError('Empty train subset')
    for episode in train + evaluation:
        scene = episode['scene_id'].replace('data/scene_datasets/', '', 1)
        if not (Path('data/scene_datasets') / scene).is_file():
            raise FileNotFoundError('Missing scene: ' + scene)
    assets = [args.baseline, args.pretrained, train_path, eval_path,
              DATASET / 'val_unseen/val_unseen_gt.json.gz',
              'data/wp_pred/check_cwp_bestdist_hfov90',
              'data/ddppo-models/gibson-2plus-resnet50.pth',
              'run_r2r/iter_train.yaml', 'run_r2r/r2r_vlnce.yaml',
              'run_r2r/e12_closed_loop_1gpu.bash']
    hashes = {str(path): sha256(path) for path in assets}
    if subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], universal_newlines=True).strip():
        raise RuntimeError('Commit tracked code changes before a reproducible E12 run')
    report = {
        'experiment': 'E12-SEG', 'utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip(),
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'train_episodes': len(train), 'eval_episodes': len(evaluation),
        'seed': get_config('run_r2r/iter_train.yaml').TASK_CONFIG.SEED,
        'asset_sha256': hashes,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('E12_PREFLIGHT_OK ' + json.dumps(report, sort_keys=True))


def check_checkpoint(args):
    import torch

    spec = importlib.util.spec_from_file_location('successor_check', ROOT / 'vlnce_baselines/successor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The declared project checkpoints include Habitat/YACS configs; use only
    # trusted local project assets (the same loader contract as E0/E11).
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    baseline = torch.load(args.baseline, map_location='cpu')
    state = {k.replace('net.module.', 'net.'): v for k, v in checkpoint['state_dict'].items()}
    base = {k.replace('net.module.', 'net.'): v for k, v in baseline['state_dict'].items()}
    prefix = 'net.vln_bert.successor.'
    decoder = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    frozen = {k: v for k, v in state.items() if not k.startswith(prefix)}
    if set(frozen) != set(base) or any(not torch.equal(frozen[k], base[k]) for k in base):
        raise RuntimeError('E12 changed frozen E0 tensors or loaded an incompatible baseline')
    expected = module.SuccessorDecoder(768, 256).state_dict()
    if set(decoder) != set(expected) or any(decoder[k].shape != expected[k].shape for k in expected):
        raise RuntimeError('E12 decoder is missing, partial or has the wrong configuration')
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise RuntimeError('E12 checkpoint contains nonfinite tensors')
    if checkpoint.get('iteration') != args.iteration or not 0 < int(decoder['updates']) <= args.iteration:
        raise RuntimeError('E12 iteration or successful optimizer-update count is wrong')
    initial = checkpoint.get('e12_initial_decoder_sha256')
    if not initial or module.decoder_digest(decoder) == initial:
        raise RuntimeError('E12 learned tensors did not change from their initial values')
    config = checkpoint['config']
    if (not config.GRPO.successor_only or config.MODEL.successor_hidden_size != 256 or
            config.GRPO.back_algo != 'control' or config.GRPO.is_requeue or
            config.TASK_CONFIG.DATASET.SPLIT != 'train' or config.TASK_CONFIG.DATASET.SUFFIX != '_10'):
        raise RuntimeError('Checkpoint training configuration differs from E12 protocol')
    print('E12_CHECKPOINT_OK iteration=%d updates=%d decoder_parameters=%d frozen_E0_equal=1 sha256=%s' % (
        args.iteration, int(decoder['updates']), sum(t.numel() for t in decoder.values() if t.is_floating_point()),
        sha256(args.checkpoint),
    ))


def results(args):
    path, episodes = dataset_episodes('val_unseen')
    expected = {str(ep['episode_id']) for ep in episodes}
    if len(expected) != 1839:
        raise ValueError('Full evaluation requires 1,839 unique episodes')
    files = list(Path(args.directory).glob('stats_ep_ckpt_*_val_unseen_r0_w1.json'))
    aggregates = list(Path(args.directory).glob('stats_ckpt_*_val_unseen.json'))
    if len(files) != 1 or len(aggregates) != 1:
        raise ValueError('Expected exactly one fresh full-evaluation result pair')
    rows = json.loads(files[0].read_text())
    report = evaluate_results(rows, expected)
    aggregate = json.loads(aggregates[0].read_text())
    for display, key in METRICS.items():
        mean = report['metrics'][display] / (1 if display == 'NE' else 100)
        if not math.isfinite(aggregate[key]) or abs(mean - aggregate[key]) > 1e-8:
            raise ValueError('Aggregate and per-episode metrics disagree: ' + key)
    report.update({'checkpoint': args.checkpoint, 'checkpoint_sha256': sha256(args.checkpoint),
                   'per_episode_file': str(files[0]), 'per_episode_sha256': sha256(files[0]),
                   'dataset_sha256': sha256(path), 'experiment': 'E12-SEG',
                   'utc': datetime.now(timezone.utc).isoformat()})
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('E12_FULL_EVAL ' + json.dumps(report, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command')
    sub.required = True
    pre = sub.add_parser('preflight')
    pre.add_argument('--baseline', required=True)
    pre.add_argument('--pretrained', required=True)
    pre.add_argument('--output', required=True)
    check = sub.add_parser('checkpoint')
    check.add_argument('checkpoint')
    check.add_argument('--baseline', required=True)
    check.add_argument('--iteration', type=int, required=True)
    result = sub.add_parser('results')
    result.add_argument('directory')
    result.add_argument('--checkpoint', required=True)
    result.add_argument('--output', required=True)
    args = parser.parse_args()
    {'preflight': preflight, 'checkpoint': check_checkpoint, 'results': results}[args.command](args)


if __name__ == '__main__':
    main()
