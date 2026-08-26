# E2: zero-initialized Gaussian position residual

E2 keeps the released ETP-R1 policy unchanged and adds one bias-free `Linear(5, 768)`
outside the pretrained position-embedding LayerNorm:

```text
position = LayerNorm(W_pos(position_7d)) + W_gauss(gaussian_5d)
```

The five graph-map features are the position standard deviation along x/y/z, the
number of merged observations, and the number of distinct frontier sources. The
new projection is zero-initialized, so enabling E2 is an exact no-op before
training. In `GRPO.gauss_only` mode, every released ETP-R1 parameter is frozen and
only the 3,840 new weights are optimized.

## Tests

The repository root is an importable Habitat plugin, so use `tests/` as pytest's
collection root when running the dependency-light E2 tests:

```bash
python -m pytest -q --rootdir=tests --import-mode=importlib tests/test_e2_gaussian.py
python tools/smoke_e2_model.py
```

The dependency-light tests cover baseline feature preservation, the zero-residual
identity, released-checkpoint compatibility, the 3,840-parameter trainable set,
and input shape validation. The model smoke then instantiates the real pretrained
ETP-R1 network and repeats the zero-output, freeze-boundary, and gradient checks.

## Single-GPU R2R-CE experiment

Run a 20-iteration smoke test from the released R2R GRPO checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e2_grpo_1gpu.bash 20 1 2334
```

If the smoke test finishes, its checkpoint loads, and the Gaussian weight has a
nonzero update, run the 500-iteration experiment:

```bash
python tools/check_e2_checkpoint.py \
  data/logs/checkpoints/e2_r2r_gaussian_residual_20/ckpt.iter20.pth

E2_EXP_NAME=e2_r2r_gaussian_residual_500 \
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e2_grpo_1gpu.bash 500 1 2334
```

Evaluate 100 episodes first, then the complete `val_unseen` split:

```bash
CKPT=data/logs/checkpoints/e2_r2r_gaussian_residual_500/ckpt.iter500.pth
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e2_eval_1gpu.bash "$CKPT" 100 4 2335
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e2_eval_1gpu.bash "$CKPT" -1 4 2335
```

The optional fifth argument scales the trained Gaussian residual without
changing the checkpoint. The default is `1.0`; `0.0` exactly disables the
residual. Scale diagnostics use separate experiment names so their result
files do not overwrite one another:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e2_eval_1gpu.bash "$CKPT" 100 4 2335 0.1
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e2_eval_1gpu.bash "$CKPT" 100 4 2336 0.25
```

Override `E2_BASE_CKPT` or `E2_PRETRAINED` only when the released assets are
stored at different paths. The scripts never download data or checkpoints.
