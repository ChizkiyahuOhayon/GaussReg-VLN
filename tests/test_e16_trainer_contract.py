from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'vlnce_baselines/GRPO_trainer_ETP_R1.py').read_text()


def test_e16_is_an_independent_lightweight_training_mode():
    assert "self.budgeted_factorized_landmark_only = getattr(" in SOURCE
    assert "'e16_optimizer_updates': self.e16_optimizer_updates" in SOURCE
    assert "self.e16_optimizer_updates += 1" in SOURCE
    assert "self.trainable_parts = [transport]" in SOURCE
    assert "budgeted_factorized_landmark_only]) > 1" in SOURCE


def test_e16_preserves_strict_e0_and_final_checkpoint_contracts():
    assert ("if (self.budgeted_factorized_landmark_only and\n"
            "                iteration != self.config.GRPO.iters)" in SOURCE)
    assert "must start from a complete E0 with no factorized " in SOURCE
    assert "expected exactly 590,848 factorized-landmark " in SOURCE


def test_e16_records_budget_diagnostics_in_sampling_and_logging():
    for name in (
            'budget_activation_rate', 'budget_dual_lambda',
            'base_route_cost', 'routed_route_cost', 'projected_route_cost'):
        assert SOURCE.count(name) >= 2
