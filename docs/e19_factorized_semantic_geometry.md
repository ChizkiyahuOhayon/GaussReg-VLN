# E19 Factorized Semantic Geometry

E19 keeps E15's verified STOP-preserving policy factorization and restores
fine local geometry only where E15 must choose among currently visible
frontiers. It is a same-host, same-data, same-budget extension of E15 and does
not use a route predictor, teacher model, future observation, goal coordinate,
or persistent geometric memory.

## Motivation and provenance

E15 is the first internal method to improve all six full R2R-CE metrics over
strict E0, but the gains are small. Its frontier router uses semantic panorama
features after graph aggregation and therefore cannot directly distinguish
candidate openings with different local 3D structure.

LCGNav reports that a local candidate-centric point-cloud encoder improves the
SPL of the same ETP-R1 host from 55.77 to 56.85. Its public implementation
constructs local point sets from candidate depth and applies a PointNet-style
encoder. E19 adopts this behavioral insight, not its source code: the module in
this repository was independently implemented and uses a smaller transient
contract tailored to E15. The inspected LCGNav repository did not expose a
top-level license at revision `60a62046639c1c8cdce91e4d69668848497d3ac0`,
so no third-party code was copied.

Primary sources:

- LCGNav paper: <https://arxiv.org/abs/2605.09053>
- LCGNav implementation: <https://github.com/shannanshouyin/LCGNav>

## Method contract

For every currently observable candidate, E19 converts its normalized depth
view to a camera-frame point set. It keeps points within 3 m, deterministically
selects 128 points with farthest-point sampling, normalizes coordinates by the
3 m radius, and encodes them with a two-layer point MLP and symmetric max pool.
The resulting residual is aligned only to current candidate graph slots.

The residual is added beside E15's instruction-conditioned landmark residual
before the frozen graph-language readout. It is never stored in `GraphMap`, so
stale local geometry cannot affect later steps. E15 still owns the policy
factorization:

```text
p_E19(STOP) = p_E0(STOP)
p_E19(frontier) = p_E0(CONTINUE) * q_E19(frontier | CONTINUE)
```

Both output projections are zero initialized. An untrained E19 is therefore
exactly E0; disabling only geometry exactly recovers E15 for the same landmark
router state. The frozen E0 backbone remains bitwise unchanged.

E19 adds 116,096 geometry parameters to E15's 590,848 landmark parameters,
for exactly 706,944 trainable parameters:

- point projection: `3 -> 128`;
- point refinement: `128 -> 128`;
- symmetric max pooling over 128 points;
- residual projection: `128 -> 768`.

## Fixed experiment

From the repository root on the server:

```bash
source /home/smbu/anaconda3/etc/profile.d/conda.sh
conda activate havlnce
export OMP_NUM_THREADS=8
bash run_r2r/e19_closed_loop_1gpu.bash 0 0 1 4 2934
```

The arguments are training GPU, evaluation GPU, training environments,
evaluation environments, and distributed base port. The runner performs
preflight, production-model smoke, independent 20- and 500-iteration training
from strict E0, checkpoint checks, and full 1,839-episode R2R-CE `val_unseen`
evaluation. It never loads E15 weights and never overwrites existing evidence.

E19 replaces E15 only if NE is lower and OSR, SR, nDTW, and SDTW are all
higher than E15, while SPL exceeds 56.85. Until that fixed evaluation finishes,
E19 has no performance claim.
