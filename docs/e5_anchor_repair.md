# E5 anchor-relative sparse repair

E5 preserves the frozen ETP-R1 action unless a learned pairwise repair policy
selects another valid graph candidate. The original E0 candidate is represented
by the same tensor index as `KEEP`; every other valid index represents
`SWITCH(candidate)`.

The frozen E0 probabilities continue to maintain ETP-R1's historical stop
ledger. This makes the zero-initialized policy trajectory-equivalent to E0,
instead of preserving only its immediate argmax actions.

Each GRPO group contains one greedy frozen-E0 anchor followed by seven
stochastic repair trajectories from the same episode. Only the repair
trajectories enter PPO/KL optimization. Their advantages are normalized reward
deltas relative to the anchor trajectory.
Waypoint augmentation is disabled so the anchor and repair samples do not
receive different random candidate proposals.

The repair head has 154,972 trainable parameters. Its output is zero-initialized,
and the `-log(K + 1)` switch prior makes initial greedy actions exactly match E0,
including tied base logits.

Run the production smoke test before training:

```bash
python tools/smoke_e5_model.py
bash run_r2r/e5_grpo_1gpu.bash 20 1 2334
python tools/check_e5_checkpoint.py \
  data/logs/checkpoints/e5_r2r_anchor_repair_20/ckpt.iter20.pth
```
