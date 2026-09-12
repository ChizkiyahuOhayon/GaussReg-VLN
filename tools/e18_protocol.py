#!/usr/bin/env python3
"""E18 fixed-recipe provenance, frozen-weight verification and paired gate."""
import argparse
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import e15_protocol as base
from tools.e16_protocol import E15_REFERENCE

E0_SHA256 = '8f90cebba7eefb9648054aa74e8c8664f23e643073ef033b99edfbe85c54f61c'
DATASET_SHA256 = '6140b46759fe332ee96aa849d4bb64e1c1829b8f65acee127355040a6ee23484'
PREFIX = 'net.vln_bert.route_attention.'


def load_checkpoint(path):
    import torch
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def normalize(state):
    return {key.replace('net.module.', 'net.'): value for key, value in state.items()}


def clone_key_mapping(frozen):
    stem = 'net.vln_bert.'
    layers = stem + 'global_encoder.encoder.x_layers.'
    indices = [int(key[len(layers):].split('.')[0]) for key in frozen if key.startswith(layers)]
    if not indices:
        raise RuntimeError('Missing E0 planning layers')
    sources = {'layer': layers + str(max(indices)) + '.',
               'query': stem + 'graph_query_text.',
               'transform': stem + 'graph_attentioned_txt_embeds_transform.',
               'head': stem + 'global_sap_head.'}
    mapping = {}
    for name, source in sources.items():
        matches = {PREFIX + name + '.' + key[len(source):]: key
                   for key in frozen if key.startswith(source)}
        if not matches:
            raise RuntimeError('Missing E0 readout: ' + name)
        mapping.update(matches)
    return mapping


def validate_state(state, frozen):
    import torch
    mapping = clone_key_mapping(frozen)
    if set(state) != set(frozen) | set(mapping):
        raise RuntimeError('E18 requires complete E0 and complete cloned branch')
    if any(not torch.equal(state[key], value) for key, value in frozen.items()):
        raise RuntimeError('E18 changed frozen E0 tensors')
    if any(state[key].shape != frozen[source].shape for key, source in mapping.items()):
        raise RuntimeError('E18 cloned shape mismatch')
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError('E18 contains nonfinite tensors')
    changed = [key for key, source in mapping.items()
               if not torch.equal(state[key], frozen[source])]
    if not changed:
        raise RuntimeError('E18 branch is unchanged from initialization')
    return {'trainable_parameters': sum(state[key].numel() for key in mapping),
            'changed_branch_tensors': len(changed), 'frozen_E0_equal': True}


def write_report(path, report):
    report['utc'] = datetime.now(timezone.utc).isoformat()
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))


def preflight(args):
    import torch
    train_path, train = base.dataset_episodes('train', '_10')
    eval_path, evaluation = base.dataset_episodes('val_unseen')
    if len(train) != 1081 or len(evaluation) != 1839 or len({str(e['episode_id']) for e in evaluation}) != 1839:
        raise ValueError('E18 requires 1,081 train and 1,839 unique evaluation episodes')
    for episode in train + evaluation:
        scene = episode['scene_id'].replace('data/scene_datasets/', '', 1)
        if not (Path('data/scene_datasets') / scene).is_file():
            raise FileNotFoundError('Missing scene: ' + scene)
    pinned = {
        str(args.baseline): E0_SHA256,
        str(args.pretrained): '203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf',
        str(train_path): 'a54947469c3afccac348d0e38645c3e4e1bd7540eb57bf947f9e427409fe9f19',
        str(eval_path): DATASET_SHA256,
        str(base.DATASET / 'val_unseen/val_unseen_gt.json.gz'): '46a2e02c4a5e3b9d3d3c936009cbe1a4e9ea8f34156589f1e69ff1ff7080c044',
        'data/wp_pred/check_cwp_bestdist_hfov90': '09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f',
        'data/ddppo-models/gibson-2plus-resnet50.pth': 'a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad',
    }
    assets = {path: base.sha256(path) for path in pinned}
    if assets != pinned:
        raise RuntimeError('Assets differ from the declared strict E0/E15 protocol')
    status = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], universal_newlines=True).strip()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], universal_newlines=True).splitlines()
    if status or any(not path.endswith('.log') for path in untracked):
        raise RuntimeError('Commit code before E18; only existing untracked .log evidence is allowed')
    for path in ('run_r2r/e18_closed_loop_1gpu.bash', 'tools/e18_protocol.py',
                 'tools/smoke_e18_model.py', 'vlnce_baselines/route_attention.py',
                 'run_r2r/iter_train.yaml', 'run_r2r/r2r_vlnce.yaml', 'docs/e18_route_attention.md'):
        assets[path] = base.sha256(path)
    write_report(args.output, dict(
        experiment='E18-ERA', git_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip(),
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), train_episodes=len(train), eval_episodes=len(evaluation),
        asset_sha256=assets, argv=sys.argv, seed=100, untracked_evidence_logs=untracked,
    ))


def check_checkpoint(args):
    from vlnce_baselines.route_attention import validate_training_config
    if base.sha256(args.baseline) != E0_SHA256:
        raise RuntimeError('Wrong strict E0 baseline')
    checkpoint, baseline = load_checkpoint(args.checkpoint), load_checkpoint(args.baseline)
    report = validate_state(normalize(checkpoint['state_dict']), normalize(baseline['state_dict']))
    validate_training_config(checkpoint['config'])
    if (checkpoint.get('iteration') != args.iteration or
            checkpoint.get('e18_optimizer_updates') != args.iteration or
            checkpoint['config'].GRPO.iters != args.iteration or
            checkpoint['config'].MODEL.route_attention_full_graph != (args.arm == 'control')):
        raise RuntimeError('E18 iteration, update count or arm mismatch')
    report.update(checkpoint_sha256=base.sha256(args.checkpoint), baseline_sha256=E0_SHA256,
                  iteration=args.iteration, optimizer_updates=checkpoint['e18_optimizer_updates'],
                  arm=args.arm, training_config=checkpoint['config'].dump())
    write_report(args.output, report)


def evaluate_results(rows, expected_ids):
    report = base.evaluate_results(rows, expected_ids)
    for key in ('success', 'oracle_success'):
        if any(row[key] not in (0, 1) for row in rows.values()):
            raise ValueError('Success metrics must be binary')
    report.pop('passes', None)
    report['thresholds'] = dict(E15_REFERENCE, SPL=56.85)
    report['e15_reference'] = E15_REFERENCE
    report['e15_passes'] = {
        key: report['metrics'][key] < value if key == 'NE' else report['metrics'][key] > value
        for key, value in E15_REFERENCE.items()}
    report['paper_spl_pass'] = report['metrics']['SPL'] > 56.85
    report['replacement_pass'] = all(report['e15_passes'].values())
    report['decision'] = 'go' if report['replacement_pass'] and report['paper_spl_pass'] else 'no-go'
    report['success_count'] = sum(row['success'] for row in rows.values())
    report['oracle_success_count'] = sum(row['oracle_success'] for row in rows.values())
    return report


def results(args):
    dataset_path, episodes = base.dataset_episodes('val_unseen')
    expected = {str(e['episode_id']) for e in episodes}
    if len(expected) != 1839 or base.sha256(dataset_path) != DATASET_SHA256:
        raise ValueError('Full evaluation requires the pinned 1,839 episodes')
    directory = Path(args.directory)
    files = list(directory.glob('stats_ep_ckpt_*_val_unseen_r0_w1.json'))
    aggregates = list(directory.glob('stats_ckpt_*_val_unseen.json'))
    if len(files) != 1 or len(aggregates) != 1:
        raise ValueError('Expected exactly one fresh full-evaluation result pair')
    rows = json.loads(files[0].read_text())
    report = evaluate_results(rows, expected)
    aggregate = json.loads(aggregates[0].read_text())
    for name, key in base.METRICS.items():
        scale = 1 if name == 'NE' else 100
        if not math.isfinite(aggregate[key]) or abs(report['metrics'][name] / scale - aggregate[key]) > 1e-8:
            raise ValueError('Aggregate/per-episode mismatch: ' + key)
    from habitat.config import Config
    evaluation = Config(new_allowed=True)
    evaluation.merge_from_file(str(directory / 'e18_eval_config.yaml'))
    if (evaluation.EVAL.EPISODE_COUNT != -1 or evaluation.EVAL.fast_eval or
            evaluation.EVAL.EPISODE_ID is not None or evaluation.EVAL.SPLIT != 'val_unseen' or
            evaluation.TASK_CONFIG.DATASET.SUFFIX != '' or evaluation.IL.back_algo != 'control' or
            not evaluation.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING or
            evaluation.MODEL.route_attention_full_graph != (args.arm == 'control') or
            not evaluation.MODEL.route_attention):
        raise ValueError('Evaluation configuration differs from the fixed E18 protocol')
    check = json.loads(Path(args.check_report).read_text())
    checkpoint_hash = base.sha256(args.checkpoint)
    if check['checkpoint_sha256'] != checkpoint_hash or check['iteration'] != 500 or check['arm'] != args.arm:
        raise ValueError('Wrong verified checkpoint report')
    report.update(experiment='E18-ERA', arm=args.arm,
                  checkpoint_sha256=checkpoint_hash, checkpoint=args.checkpoint,
                  per_episode_file=str(files[0]), per_episode_sha256=base.sha256(files[0]),
                  dataset_sha256=DATASET_SHA256, checkpoint_check=check,
                  evaluation_config=evaluation.dump())
    write_report(args.output, report)


def compare(args):
    control = json.loads(Path(args.control).read_text())
    route = json.loads(Path(args.route).read_text())
    if (control['arm'] != 'control' or route['arm'] != 'route' or
            control['dataset_sha256'] != route['dataset_sha256'] or
            control['episodes'] != 1839 or route['episodes'] != 1839 or
            control['checkpoint_check']['trainable_parameters'] != route['checkpoint_check']['trainable_parameters']):
        raise ValueError('Unmatched E18 experiment arms')
    passes = {key: route['metrics'][key] < value if key == 'NE' else route['metrics'][key] > value
              for key, value in control['metrics'].items()}
    changes = {}
    per_episode = []
    for report in (control, route):
        if base.sha256(report['per_episode_file']) != report['per_episode_sha256']:
            raise ValueError('Changed per-episode evidence')
        per_episode.append(json.loads(Path(report['per_episode_file']).read_text()))
    old, new = per_episode
    if set(old) != set(new):
        raise ValueError('Unmatched paired episode identities')
    for key in ('success', 'oracle_success'):
        changes[key] = {'recovered': [ep for ep in old if new[ep][key] > old[ep][key]],
                        'lost': [ep for ep in old if new[ep][key] < old[ep][key]]}
    write_report(args.output, dict(
        experiment='E18-ERA', control=args.control, route=args.route,
        route_metrics=route['metrics'], control_metrics=control['metrics'],
        delta_route_minus_control={key: route['metrics'][key] - value for key, value in control['metrics'].items()},
        control_passes=passes, paired_changes=changes,
        decision='go' if route['decision'] == 'go' and all(passes.values()) else 'no-go',
        reason='Requires six strict improvements over E15 and matched control, plus SPL >56.85',
    ))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('preflight')
    pre.add_argument('--baseline', required=True)
    pre.add_argument('--pretrained', required=True)
    check = sub.add_parser('checkpoint')
    check.add_argument('checkpoint')
    check.add_argument('--baseline', required=True)
    check.add_argument('--iteration', type=int, choices=(20, 500), required=True)
    check.add_argument('--arm', choices=('control', 'route'), required=True)
    result = sub.add_parser('results')
    result.add_argument('directory')
    result.add_argument('--checkpoint', required=True)
    result.add_argument('--check-report', required=True)
    result.add_argument('--arm', choices=('control', 'route'), required=True)
    paired = sub.add_parser('compare')
    paired.add_argument('--control', required=True)
    paired.add_argument('--route', required=True)
    for command in (pre, check, result, paired):
        command.add_argument('--output', required=True)
    args = parser.parse_args()
    {'preflight': preflight, 'checkpoint': check_checkpoint,
     'results': results, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
