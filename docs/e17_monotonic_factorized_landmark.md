# E17: Monotonic Factorized Landmark Routing

E17 keeps E15's frozen E0 STOP decision and conditional frontier policy. It
adds a parameter-free monotonic alignment between real visited observations and
the four ordered instruction slots. The resulting stage posterior weights the
existing E15 frontier-slot memory before action scoring.

## Code-to-paper map

| Code | Paper term |
|---|---|
| `monotonic_progress` | Monotonic Instruction Alignment |
| `MonotonicLandmarkTransport` | Stage-Conditioned Landmark Router |
| `factorized_action_log_probs` | Factorized STOP/Frontier Policy |

The model introduces no teacher, future observation decoder, goal coordinate,
route-cost penalty, or new trainable parameter beyond E15's 590,848-parameter
transport. Setting `MODEL.factorized_landmark_monotonic=False` recovers E15;
setting `MODEL.factorized_landmark_size=0` recovers E0.

The production entry point is `run_r2r/e17_closed_loop_1gpu.bash`. It runs the
production smoke, an independent 20-iteration check, an independent
500-iteration training run from strict E0, checkpoint validation, and the full
1,839-episode R2R-CE `val_unseen` evaluation.
