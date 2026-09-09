# E14 Landmark Transport Graph

E14 tests whether instruction-conditioned visual content should enter the
graph before ETP-R1's global graph encoder. It starts from the strict E0
checkpoint and does not compose with E2--E13.

## Method contract

For each graph entity, `GraphMap` retains the frozen panorama or candidate
embeddings that existed before entity-level mean pooling. The instruction is
partitioned into four ordered contiguous slots. Two learned projections map
the source views and instruction slots to 128 dimensions; masked compatibility
weights transport their elementwise content interaction into four slot
memories. A learned projection maps the resulting 512-dimensional memory to a
768-dimensional residual that is added to the graph token before the frozen
global graph encoder.

The residual output is zero initialized, so enabling an untrained E14 module
is exactly equivalent to E0. Replay stores frozen source-view embeddings on
CPU and recomputes both learned projections, allowing the policy objective to
train the complete E14 path. E0 inputs and parameters remain detached.

The production configuration has exactly 590,848 trainable parameters:

- view projection: `768 -> 128`;
- instruction-slot projection: `768 -> 128`;
- transport output: `4 * 128 -> 768`.

## Fixed experiment

Run the complete protocol from the repository root:

```bash
source /home/smbu/anaconda3/etc/profile.d/conda.sh
conda activate havlnce
bash run_r2r/e14_closed_loop_1gpu.bash 1 2 1 4 2634
```

The arguments are training GPU, evaluation GPU, training environments,
evaluation environments, and base distributed port. The runner executes:

1. asset and revision preflight;
2. production-model E0/load/gradient smoke;
3. independent 20-iteration training and checkpoint validation;
4. independent 500-iteration training from E0 and checkpoint validation;
5. one 1,839-episode R2R-CE `val_unseen` evaluation and six-metric gate.

Each training arm writes only its final checkpoint. This bounds checkpoint
storage to one 20-iteration file and one 500-iteration file.

The fixed training recipe uses eight rollouts, one update epoch, AdamW learning
rate `1e-4`, KL coefficient `0.04`, no AMP/dropout/waypoint augmentation, and
the `control` back algorithm. No E13 or 20-iteration checkpoint is reused.

Evidence is written under `data/logs/e14_r2r_landmark_transport/`. The final
machine-readable result is `full_result.json`; `exit_status.txt` identifies the
last completed phase. Existing evidence is never overwritten. Set a new
alphanumeric `E14_RUN_ID` only for an infrastructure retry.

The result is a GO only if all fixed conditions hold simultaneously: NE
`<3.93`, OSR `>71.45`, SR `>65.31`, SPL `>56.85`, nDTW `>65.8165`, and SDTW
`>53.9994`. Until that evaluation completes, E14 has no performance claim.
