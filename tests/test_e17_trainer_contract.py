from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'vlnce_baselines/GRPO_trainer_ETP_R1.py').read_text()


def test_e17_is_an_independent_lightweight_training_mode():
    assert "self.monotonic_factorized_landmark_only = getattr(" in SOURCE
    assert "'e17_optimizer_updates': self.e17_optimizer_updates" in SOURCE
    assert 'self.e17_optimizer_updates += 1' in SOURCE
    assert 'self.trainable_parts = [transport]' in SOURCE
    assert 'if (monotonic_factorized_landmark_only and any([' in SOURCE


def test_e17_preserves_e0_and_final_checkpoint_contracts():
    assert ("if (self.monotonic_factorized_landmark_only and\n"
            "                iteration != self.config.GRPO.iters)" in SOURCE)
    assert 'must start from a complete E0 with no factorized ' in SOURCE
    assert 'expected exactly 590,848 factorized-landmark ' in SOURCE


def test_e17_records_stage_diagnostics_in_sampling_and_logging():
    for name in (
            'instruction_stage_expected', 'instruction_stage_entropy',
            'instruction_stage_advance_mass'):
        assert SOURCE.count(name) >= 2


def test_e17_rejects_nonfinite_gradient_before_optimizer_step():
    assert 'error_if_nonfinite=(' in SOURCE
    assert 'self.monotonic_factorized_landmark_only' in SOURCE
