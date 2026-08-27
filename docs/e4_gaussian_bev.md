# E4 Gaussian-BEV residual

E4 keeps the released ETP-R1 policy frozen and learns a 50,619-parameter
egocentric Gaussian field over graph-map tokens. The output layer is zero
initialized, so enabling E4 initially preserves every E0 navigation logit.

## Local checks

```bash
python -m pytest -q --rootdir=tests --import-mode=importlib \
  tests/test_e2_gaussian.py
python tools/smoke_e4_model.py
```

## A40 smoke training

```bash
bash run_r2r/e4_grpo_1gpu.bash 20 1 2334 2>&1 | tee e4_grpo_smoke20.log
python tools/check_e4_checkpoint.py \
  data/logs/checkpoints/e4_r2r_gaussian_bev_20/ckpt.iter20.pth
```

After the 20-iteration loop passes, run 500 iterations and evaluate all 1,839
R2R-CE `val_unseen` episodes:

```bash
bash run_r2r/e4_grpo_1gpu.bash 500 1 2334 2>&1 | tee e4_grpo_500.log
CKPT=data/logs/checkpoints/e4_r2r_gaussian_bev_500/ckpt.iter500.pth
bash run_r2r/e4_eval_1gpu.bash "$CKPT" -1 4 2335 2>&1 | tee e4_eval_full.log
```
