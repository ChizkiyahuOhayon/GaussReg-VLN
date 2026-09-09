# E15 Factorized Landmark Routing

E15 isolates the useful part of E14—landmark-conditioned frontier ranking—from
its observed failure mode: changing the competition between STOP and
navigation. It starts from the strict E0 checkpoint and does not compose with
E2--E14.

## Method contract

The frozen E0 policy retains complete authority over whether to stop. E15 only
models the conditional distribution over valid unvisited frontiers:

```
p_E15(STOP) = p_E0(STOP)
p_E15(frontier) = p_E0(CONTINUE) * q_E15(frontier | CONTINUE)
```

For greedy inference, an E0 STOP remains STOP; otherwise E15 chooses the
highest-scoring valid frontier. Thus landmark routing cannot create or suppress
a terminal decision. With a zero-initialized output projection, `q_E15` equals
E0's conditional frontier distribution, making the untrained model equivalent
to E0.

`GraphMap` retains frozen pre-pooling panorama and candidate embeddings. The
same four-slot instruction-conditioned transport used in E14 predicts a
frontier representation residual, but E15 applies it after the frozen global
graph encoder and only to unvisited frontier tokens. E0 inputs and parameters
remain detached.

The production module has exactly 590,848 trainable parameters:

- view projection: `768 -> 128`;
- instruction-slot projection: `768 -> 128`;
- transport output: `4 * 128 -> 768`.

## Fixed experiment

Run the complete protocol from the repository root:

```bash
source /home/smbu/anaconda3/etc/profile.d/conda.sh
conda activate havlnce
bash run_r2r/e15_closed_loop_1gpu.bash 1 2 1 4 2834
```

The arguments are training GPU, evaluation GPU, training environments,
evaluation environments, and base distributed port. The runner performs asset
and revision preflight, a production-model factorization smoke, independent
20- and 500-iteration training runs from E0, strict checkpoint checks, and one
1,839-episode R2R-CE `val_unseen` evaluation.

Each training arm writes only its final checkpoint. The fixed recipe uses eight
rollouts, one update epoch, AdamW learning rate `1e-4`, KL coefficient `0.04`,
no AMP/dropout/waypoint augmentation, and the `control` back algorithm. E14 and
the 20-iteration checkpoint are not reused.

Evidence is written under `data/logs/e15_r2r_factorized_landmark/`. Existing
evidence is never overwritten. Set a new alphanumeric `E15_RUN_ID` only for an
infrastructure retry; every retry still starts from E0.

The result is a GO only if all fixed conditions hold simultaneously: NE
`<3.93`, OSR `>71.45`, SR `>65.31`, SPL `>56.85`, nDTW `>65.8165`, and SDTW
`>53.9994`. Until the full evaluation completes, E15 has no performance claim.
