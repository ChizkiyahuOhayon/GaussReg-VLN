"""Test E15 orchestration; stubs do not replace Habitat validation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_e15_only_writes_the_final_training_checkpoint():
    source = (ROOT / 'vlnce_baselines/GRPO_trainer_ETP_R1.py').read_text()
    assert ('self.factorized_landmark_only and\n'
            '                iteration != self.config.GRPO.iters' in source)


def _run(tmp_path, fail20=False):
    (tmp_path / 'run_r2r').mkdir()
    shutil.copyfile(ROOT / 'run_r2r/e15_closed_loop_1gpu.bash',
                    tmp_path / 'run_r2r/e15_closed_loop_1gpu.bash')
    (tmp_path / 'run_r2r/habitat_env.bash').write_text('return 0\n')
    (tmp_path / 'bin').mkdir()
    python = tmp_path / 'bin/python'
    python.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
with open('commands.jsonl', 'a') as stream:
    stream.write(json.dumps({'argv': sys.argv[1:],
                             'gpu': os.getenv('CUDA_VISIBLE_DEVICES')}) + '\\n')
if os.getenv('FAIL20') and 'GRPO.iters' in sys.argv:
    if sys.argv[sys.argv.index('GRPO.iters') + 1] == '20':
        sys.exit(17)
''')
    python.chmod(0o755)
    env = dict(os.environ,
               PATH=str(tmp_path / 'bin') + os.pathsep + os.environ['PATH'])
    env.pop('E15_RUN_ID', None)
    if fail20:
        env['FAIL20'] = '1'
    process = subprocess.run(
        ['bash', 'run_r2r/e15_closed_loop_1gpu.bash',
         '1', '2', '1', '4', '2834'],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    calls = [json.loads(line) for line in
             (tmp_path / 'commands.jsonl').read_text().splitlines()]
    return process, calls


def test_runner_uses_fixed_contract_and_independent_training(tmp_path):
    process, calls = _run(tmp_path)
    assert process.returncode == 0, process.stderr
    training = [call for call in calls if 'GRPO.iters' in call['argv']]
    assert len(training) == 2
    for call, iterations in zip(training, ['20', '500']):
        argv = call['argv']
        assert argv[argv.index('GRPO.iters') + 1] == iterations
        assert argv[argv.index('GRPO.factorized_landmark_only') + 1] == 'True'
        assert argv[argv.index('GRPO.landmark_transport_only') + 1] == 'False'
        assert argv[argv.index('MODEL.factorized_landmark_size') + 1] == '128'
        assert argv[argv.index('MODEL.landmark_transport_size') + 1] == '0'
        assert argv[argv.index('GRPO.ckpt_to_load') + 1].endswith(
            'release_r2r_grpo/store/ckpt.iter270.pth'
        )
        assert call['gpu'] == '1'
    evaluation = next(call for call in calls
                      if 'EVAL.EPISODE_COUNT' in call['argv'])
    assert evaluation['gpu'] == '2'
    assert evaluation['argv'][
        evaluation['argv'].index('EVAL.EPISODE_COUNT') + 1
    ] == '-1'
    gate = next(call for call in calls
                if call['argv'][:2] == ['tools/e15_protocol.py', 'results'])
    assert gate['argv'][gate['argv'].index('--output') + 1].endswith(
        'e15_r2r_factorized_landmark/full_result.json'
    )
    status = tmp_path / 'data/logs/e15_r2r_factorized_landmark/exit_status.txt'
    assert 'phase=completed exit=0' in status.read_text()


def test_smoke_failure_stops_before_main_training_and_evaluation(tmp_path):
    process, calls = _run(tmp_path, fail20=True)
    assert process.returncode == 17
    assert len([call for call in calls if 'GRPO.iters' in call['argv']]) == 1
    assert not any('EVAL.EPISODE_COUNT' in call['argv'] for call in calls)
    status = tmp_path / 'data/logs/e15_r2r_factorized_landmark/exit_status.txt'
    assert 'phase=train20 exit=17' in status.read_text()
