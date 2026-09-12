"""Execute the shell orchestration with phase-aware subprocess stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASES = ['preflight', 'model_smoke'] + [arm + '_' + phase for arm in ('control', 'route')
          for phase in ('train20', 'check20', 'train500', 'check500', 'full_eval', 'gate')] + ['compare']


def run(tmp_path, fail=None):
    (tmp_path / 'run_r2r').mkdir()
    shutil.copyfile(ROOT / 'run_r2r/e18_closed_loop_1gpu.bash', tmp_path / 'run_r2r/e18_closed_loop_1gpu.bash')
    (tmp_path / 'run_r2r/habitat_env.bash').write_text('return 0\n')
    (tmp_path / 'bin').mkdir()
    stub = tmp_path / 'bin/python'
    stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
phase = os.getenv('E18_PHASE')
with open('calls.jsonl', 'a') as stream:
    stream.write(json.dumps(dict(argv=sys.argv[1:], phase=phase,
        gpu=os.getenv('CUDA_VISIBLE_DEVICES'))) + '\\n')
if phase == os.getenv('FAIL_PHASE'):
    sys.exit(17)
# A valid performance no-go is a successful execution, as in the real gate.
print('{"decision": "no-go"}')
''')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path / 'bin') + os.pathsep + os.environ['PATH'])
    env.pop('E18_RUN_ID', None)
    if fail:
        env['FAIL_PHASE'] = fail
    process = subprocess.run(['bash', 'run_r2r/e18_closed_loop_1gpu.bash', '3', '2', '1', '4', '3138'],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    return process, calls, env


def test_both_arms_independent_training_and_full_evaluation(tmp_path):
    result, calls, env = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert [call['phase'] for call in calls] == PHASES
    training = [call for call in calls if 'GRPO.iters' in call['argv']]
    assert len(training) == 4
    for call, iterations in zip(training, ['20', '500', '20', '500']):
        args = call['argv']
        assert args[args.index('GRPO.iters') + 1] == iterations
        assert args[args.index('GRPO.ckpt_to_load') + 1].endswith('release_r2r_grpo/store/ckpt.iter270.pth')
        assert args[args.index('GRPO.lr') + 1] == '0.00002'
        assert args[args.index('MODEL.factorized_landmark_size') + 1] == '0'
        assert args[args.index('MODEL.route_attention_full_graph') + 1] == ('True' if call['phase'].startswith('control') else 'False')
        assert call['gpu'] == '3'
    for call in calls:
        if 'EVAL.EPISODE_COUNT' in call['argv']:
            args = call['argv']
            assert args[args.index('EVAL.EPISODE_COUNT') + 1] == '-1'
            assert args[args.index('EVAL.fast_eval') + 1] == 'False'
            assert call['gpu'] == '2'
    status = tmp_path / 'data/logs/e18_r2r_route_attention/exit_status.txt'
    assert status.read_text().strip() == 'phase=completed exit=0'
    original = status.read_text()
    repeated = subprocess.run(['bash', 'run_r2r/e18_closed_loop_1gpu.bash'], cwd=tmp_path,
                              env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert 'Refusing to overwrite' in repeated.stderr
    assert status.read_text() == original


@pytest.mark.parametrize('phase', PHASES)
def test_any_failure_stops_at_its_phase(tmp_path, phase):
    result, calls, _ = run(tmp_path, phase)
    assert result.returncode == 17, result.stderr
    assert [call['phase'] for call in calls] == PHASES[:PHASES.index(phase) + 1]
    status = tmp_path / 'data/logs/e18_r2r_route_attention/exit_status.txt'
    assert status.read_text().strip() == 'phase=%s exit=17' % phase
