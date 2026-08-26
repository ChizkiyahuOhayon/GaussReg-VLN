# E3: lightweight candidate residual scorer

E3 starts from the released ETP-R1 GRPO checkpoint. The original policy remains
frozen. A shared two-layer MLP adds one bounded residual logit to each graph
candidate and STOP using the detached language-conditioned candidate
representation, relative-position/Gaussian features, visited state, and an
explicit STOP indicator.

The output layer is zero-initialized. Before training, E3 therefore produces
exactly the baseline logits. E2's Gaussian position residual is disabled in E3;
the five Gaussian statistics are inputs to the scorer rather than a direct
position-embedding perturbation.

## Checks

```bash
python tools/smoke_e3_model.py
python -m pytest -q tests/test_e2_gaussian.py
```

The smoke check requires the joint-pretraining checkpoint. The unit tests do
not require Habitat or model assets.

## Single-GPU R2R-CE experiment

Run the 20-iteration integration check first:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e3_grpo_1gpu.bash 20 1 2334
```

Inspect its learned output layer:

```bash
python tools/check_e3_checkpoint.py \
  data/logs/checkpoints/e3_r2r_candidate_scorer_20/ckpt.iter20.pth
```

Then run 500 iterations and the paired 100-episode evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e3_grpo_1gpu.bash 500 1 2334
CKPT=data/logs/checkpoints/e3_r2r_candidate_scorer_500/ckpt.iter500.pth
CUDA_VISIBLE_DEVICES=0 bash run_r2r/e3_eval_1gpu.bash "$CKPT" 100 4 2335
```

Use `-1` instead of `100` only after the paired gate improves over the E0
control. Override `E3_BASE_CKPT` or `E3_PRETRAINED` only when the released
assets are stored elsewhere.
