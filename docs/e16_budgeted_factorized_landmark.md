# E16 Budgeted Factorized Landmark Routing

E16 retains E15's factorized landmark representation and frozen STOP decision,
then constrains frontier rerouting by the graph cost already available to E0.
It starts independently from the strict E0 checkpoint and does not modify or
reuse the E15 checkpoint.

## Method contract

Let `q0` be E0's conditional distribution over valid unvisited frontiers and
`q15` the landmark-routed distribution. E16 computes the minimum-KL projection

```
q16 = argmin_q KL(q || q15)
      subject to E_q[c] <= E_q0[c]
```

where `c` is shortest-path graph distance from the latest visited node. The
closed-form family is `q16(a) proportional to q15(a) exp(-lambda * c(a))`; a
fixed bisection solves the single nonnegative dual variable. If E15 already
meets the E0 budget, the projection is an exact no-op.

E0 retains the complete STOP probability and decision. The constraint acts
only on valid frontier logits, adds no learned parameters, and uses no teacher,
cost head, or tunable loss coefficient. The only trainable component remains
E15's 590,848-parameter factorized landmark router. Its output is zero
initialized, so untrained E16 is exactly E0.

## Fixed experiment

From the repository root:

```bash
source /home/smbu/anaconda3/etc/profile.d/conda.sh
conda activate havlnce
bash run_r2r/e16_closed_loop_1gpu.bash 1 2 1 4 2934
```

Arguments are training GPU, evaluation GPU, training environments, evaluation
environments, and base distributed port. The runner performs provenance and
asset preflight, a production-model smoke, independent 20- and 500-iteration
training from E0, strict checkpoint validation, and full 1,839-episode R2R-CE
`val_unseen` evaluation.

Evidence is written under
`data/logs/e16_r2r_budgeted_factorized_landmark/`. Existing evidence is never
overwritten. Set a new alphanumeric `E16_RUN_ID` only for an infrastructure
retry; retries still start from E0. E15 code, evidence, and checkpoints remain
untouched.

The fixed GO gate requires all metrics simultaneously: NE `<3.93`, OSR
`>71.45`, SR `>65.31`, SPL `>56.85`, nDTW `>65.8165`, and SDTW `>53.9994`.
Promotion also requires strict improvement over E15 on each of the same six
metrics. Until full evaluation completes, E16 has no performance claim.
