"""Test orchestration only; these stubs never stand in for Habitat validation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, fail20=False):
    (tmp_path / 'run_r2r').mkdir()
    shutil.copyfile(ROOT / 'run_r2r/e12_closed_loop_1gpu.bash',
                    tmp_path / 'run_r2r/e12_closed_loop_1gpu.bash')
    (tmp_path / 'run_r2r/habitat_env.bash').write_text('return 0\n')
    (tmp_path / 'bin').mkdir()
    python = tmp_path / 'bin/python'
    python.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
with open('commands.jsonl', 'a') as stream:
    stream.write(json.dumps({'argv': sys.argv[1:], 'gpu': os.getenv('CUDA_VISIBLE_DEVICES')}) + '\\n')
if os.getenv('FAIL20') and 'GRPO.iters' in sys.argv:
    if sys.argv[sys.argv.index('GRPO.iters') + 1] == '20':
        sys.exit(17)
''')
    python.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path / 'bin') + os.pathsep + os.environ['PATH'])
    env.pop('E12_RUN_ID', None)
    if fail20:
        env['FAIL20'] = '1'
    process = subprocess.run(['bash', 'run_r2r/e12_closed_loop_1gpu.bash', '1', '2', '1', '4', '2434'],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in (tmp_path / 'commands.jsonl').read_text().splitlines()]
    return process, calls


def test_runner_independently_trains_and_evaluates_full_set_on_requested_gpus(tmp_path):
    process, calls = _run(tmp_path)
    assert process.returncode == 0, process.stderr
    train = [call for call in calls if 'GRPO.iters' in call['argv']]
    assert len(train) == 2
    for call, iterations in zip(train, ['20', '500']):
        argv = call['argv']
        assert argv[argv.index('GRPO.iters') + 1] == iterations
        assert argv[argv.index('GRPO.ckpt_to_load') + 1].endswith('release_r2r_grpo/store/ckpt.iter270.pth')
        assert argv[argv.index('GRPO.is_requeue') + 1] == 'False'
        assert call['gpu'] == '1'
    evaluation = next(call for call in calls if 'EVAL.EPISODE_COUNT' in call['argv'])
    assert evaluation['gpu'] == '2'
    argv = evaluation['argv']
    assert argv[argv.index('EVAL.EPISODE_COUNT') + 1] == '-1'
    assert argv[argv.index('EVAL.CKPT_PATH_DIR') + 1].endswith('_500/ckpt.iter500.pth')
    assert 'phase=completed exit=0' in (tmp_path / 'data/logs/e12_r2r_successor/exit_status.txt').read_text()


def test_smoke_training_failure_prevents_main_training_and_evaluation(tmp_path):
    process, calls = _run(tmp_path, fail20=True)
    assert process.returncode == 17
    assert len([call for call in calls if 'GRPO.iters' in call['argv']]) == 1
    assert not any('EVAL.EPISODE_COUNT' in call['argv'] for call in calls)
    assert 'phase=train20 exit=17' in (tmp_path / 'data/logs/e12_r2r_successor/exit_status.txt').read_text()
