# Iteration Experiments

Experiments on test-time iteration scaling, adaptive stopping, and confound isolation.

See the main [EXPERIMENTS.md](../EXPERIMENTS.md) for full results.

## Key Files

- `eval_more_iters.py` - Test model at different iteration counts (no retraining)
- `eval_confidence_stop.py` - Confidence-based and oscillation-based adaptive stopping
- `eval_fixed_point.py` - Test if correct solution is a stable fixed point of f
- `exp_baseline_lr2e3.py` - LR=2e-3 (**SOTA: 98.9%** at 1024 test iters)
- `exp_bs2048_baseline.py` - BS=2048, 16-iter, reverse curriculum (prev SOTA: 98.2%)
- `exp_bs2048_mixed.py` - BS=2048, 16-iter, mixed sampling (isolates curriculum effect)
- `exp_bs1024_curriculum.py` - BS=1024, 16-iter, reverse curriculum (tests smaller batch)
- `exp_32iters.py` - BS=2048, 32-iter, mixed sampling
- `exp_32iters_curriculum.py` - BS=2048, 32-iter, reverse curriculum (isolates training iter effect)
- `exp_bs2048_100k.py` - BS=2048, 100K steps with stretched LR (negative result)
- `exp_bs2048_fixedpoint.py` - FP: 2x CE weight on correct cells (negative result)
- `exp_bs2048_fp_l2.py` - FP: L2 loss toward target on correct cells (negative result)
- `exp_bs2048_fp_copy.py` - FP: self-consistency copy loss (negative result)
- `exp_bs2048_fp_gradmask.py` - FP: gradient masking on correct cells (negative result)
- `exp_wider_6h.py` - d_model=192, 6 heads (tests wider model)
- `exp_wider_6h_lr2e3.py` - d_model=192, LR=2e-3 (peaks at 64 iters, collapses at 128)
- `exp_wider_6h_lowlr.py` - d_model=192, LR=1e-3 (tests if lower LR fixes wider collapse)
- `exp_smaller_3h.py` - d_model=96, 3 heads (tests smaller model)
- `exp_baseline_lr2e3.py` - LR=2e-3 (**NEW SOTA: 98.9%** at 1024 test iters)
- `exp_baseline_lr25e4.py` - LR=2.5e-3 (collapses at 128 iters)
- `exp_baseline_lr3e3.py` - LR=3e-3 (collapses at 64 iters)
- `exp_baseline_lr1e3.py` - LR=1e-3 (collapses at 128 iters)
- `exp_3phase_40k.py` - 3-phase curriculum, 40K steps (stable but lower ceiling)
- `exp_3phase_50k.py` - 3-phase curriculum, 50K steps (collapses)
- `exp_qhead.py` - Q-head learned halt signal (16 iters, negative result)
- `exp_qhead_32.py` - Q-head learned halt signal (32 iters, negative result)
- `eval_interventions.py` - Test-time interventions (damping, pred scaling, pre-norm)
- `eval_spectral_radius.py` - Jacobian spectral radius via power iteration
- `modal_eval_interventions.py` - Modal wrapper for intervention sweeps
- `modal_spectral_stable.py` - Modal wrapper for spectral radius + stable model interventions
- `modal_eval.py` - Modal wrapper for running eval_more_iters on GPU

## Test-Time Iteration Scaling

Accuracy at various test-time iteration counts:

| Model | Notes | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| **exp_baseline_lr2e3** | **LR=2e-3 (NEW SOTA)** | **81.8%** | **88.5%** | **92.5%** | **95.3%** | **97.3%** | **98.5%** | **98.9%** | **98.8%** |
| exp_bs2048_baseline | LR=1.5e-3 (prev SOTA) | 81.4% | 88.1% | 92.4% | 94.9% | 96.6% | 97.5% | 98.1% | 98.2% |
| exp_baseline_lr25e4 | LR=2.5e-3 | 81.8% | 87.7% | 90.7% | 84.1% | 50.5% | 4.5% | 0.1% | — |
| exp_baseline_lr3e3 | LR=3e-3 | 82.1% | 88.5% | 39.9% | 7.1% | 1.7% | 0.9% | 0.4% | — |
| exp_baseline_lr1e3 | LR=1e-3 | 80.9% | 87.2% | 90.9% | 89.8% | 80.2% | 50.2% | 20.8% | 5.3% |
| exp_wider_6h_lr2e3 | d=192, LR=2e-3 | 84.4% | 90.8% | 94.3% | 85.9% | 23.3% | 6.9% | 3.5% | — |
| exp_wider_6h | d=192, LR=1.5e-3 | 84.7% | 91.3% | 64.2% | 28.2% | 11.5% | 4.5% | 1.1% | — |
| exp_wider_6h_lowlr | d=192, LR=1e-3 | 84.1% | 90.8% | 94.5% | 94.7% | 8.2% | 0.9% | 0.1% | 0.0% |
| exp_smaller_3h | d=96, 3 heads | 76.8% | 83.0% | 86.5% | 87.8% | 87.7% | 87.4% | 86.4% | 73.2% |
| exp_3phase_40k | 3-phase, 10K/10K/20K | 80.0% | 86.8% | 91.0% | 94.0% | 95.4% | 95.9% | 95.9% | 95.9% |
| exp_3phase_50k | 3-phase, 15K/15K/20K | 81.2% | 87.6% | 34.9% | 6.2% | 3.5% | 0.7% | 0.0% | — |
| exp_faster_2drope | BS=4096 | 82.5% | 88.4% | 20.8% | — | — | — | — | — |
| exp_bs2048_mixed | mixed sampling | 81.2% | 88.3% | 92.3% | 94.9% | 96.4% | 97.0% | 97.3% | — |
| exp_bs1024_curriculum | BS=1024 | 79.7% | 85.9% | 89.5% | 91.6% | 54.8% | 1.2% | 0.0% | — |
| exp_32iters | 32-iter, mixed | 72.4% | 83.2% | 88.7% | — | 91.8% | 88.3% | — | — |
| exp_32iters_curriculum | 32-iter | — | 83.4% | 89.6% | 78.7% | 61.6% | 47.3% | 20.7% | — |
| exp_bs2048_100k | 100K steps (stretched LR) | 83.1% | 89.5% | 74.5% | 9.2% | 3.2% | 1.5% | 0.5% | — |
| exp_bs2048_fixedpoint | FP: 2x CE on correct | 80.8% | 86.9% | 90.5% | 92.9% | 94.0% | 71.5% | 39.4% | — |
| exp_bs2048_fp_l2 | FP: L2 toward target | 80.5% | 86.9% | 90.6% | 89.3% | 19.1% | 16.2% | 14.9% | — |
| exp_bs2048_fp_copy | FP: self-consistency | 81.7% | 88.1% | 90.9% | 92.5% | 93.1% | 93.3% | 83.7% | — |
| exp_bs2048_fp_gradmask | FP: zero CE on correct | 3.2% | — | — | — | — | — | — | — |

## Other Results

| Experiment | Accuracy | Notes |
|---|---|---|
| BS=4096 + oscillation stop | 91.1% | Causal, deployable |
| BS=4096 + peak confidence (oracle) | 91.5% | Requires all iters retroactively |
| Q-head (16 iters) | 79.4% | Loss competition hurts main task |
| Q-head (32 iters) | 78.4% | Same issue |

## Fixed-Point Analysis

### Teacher Forcing — Disproven

Both test-time and training-time teacher forcing were tested. Neither works.

**Cold-start fixed-point test:** Feed the correct solution (as one-hot softmax predictions) into f with a cold hidden state (h_prev = initial_encoder(puzzle)). (`eval_fixed_point.py`)

| Model | Cells preserved | Puzzles perfectly preserved |
|---|---|---|
| BS=4096 baseline (16-iter) | 14.4% | 0/25000 |
| 32-iter mixed | 47.6% | 0/25000 |

f destroys the correct solution when starting from a cold hidden state. But this test is misleading — during normal inference, h_prev is warm after hundreds of iterations.

**Warm-state fixed-point test:** Run model for 1022–1026 iterations and compare outputs at each step. Result: 24513/25000 puzzles solved identically at every iteration. The model converges to a perfect argmax-fixed-point — once h_prev is warm, f preserves its own predictions exactly.

**Training-time teacher forcing (32-iter training):** Train with 32 iterations so easy puzzles are solved by iter ~10, giving f 20+ iterations of "correct input → stay correct" signal.

Both 32-iter models (mixed and curriculum) collapse past 128-256 test iters, while 16-iter BS=2048 models scale to 2048+. The implicit teacher forcing signal is too weak — f still learns to re-solve rather than preserve.

**Conclusion:** The cold-start test disproved teacher forcing (f can't preserve solutions it didn't derive itself), but the warm-state test shows f naturally converges to an argmax-fixed-point. Fixed-point behavior is emergent — no explicit loss needed.

### Explicit Fixed-Point Losses — All Hurt

Four variants tested, all degrade iteration scaling. The stronger the pressure, the earlier the collapse:

1. **Preservation weighting** (exp_bs2048_fixedpoint) — 2x CE weight on cells correct at previous iteration. Collapses at 512 iters.
2. **L2 toward target** (exp_bs2048_fp_l2) — MSE between softmax and one-hot target on correct cells. Collapses at 128 iters.
3. **Self-consistency copy** (exp_bs2048_fp_copy) — MSE between softmax at iter t and t-1 on correct cells. Gentlest; collapses at 1024 iters (83.7%).
4. **Gradient masking** (exp_bs2048_fp_gradmask) — zero CE on correct cells. Catastrophic failure (3.2%).

The baseline with no fixed-point loss reaches 98.1% at 1024 iters. Every explicit fixed-point intervention disrupts the flat minimum that enables stable iteration scaling.

### 100K Training Steps — LR Schedule Confound

Training for 100K steps with cosine decay stretched over 100K (exp_bs2048_100k) collapses at 64 test iters. The step-50K checkpoint already collapses — confirming the cause is the stretched LR schedule (LR still high at step 50K), not overtraining. The baseline's 50K cosine fully anneals by training end, enabling the flat minimum.

## Conditions for Stable Iteration Scaling

Stable test-time iteration scaling (more iters → monotonically better accuracy) requires all of the following:

1. **LR in a narrow band** — for d=128, only LR=1.5e-3 to 2e-3 works. Both higher (2.5e-3, 3e-3) and lower (1e-3) collapse. The optimum is sharp at 2e-3.
2. **BS=2048** — BS=4096 collapses at 48 iters (too little gradient noise → sharp minima), BS=1024 at 256 (too much noise). The right noise level finds flat minima where f is naturally contractive.
3. **Small enough model** — d=128 scales to 1024+. d=192 collapses at every LR tested. d=96 peaks early and slowly degrades. More capacity makes the iteration map harder to keep contractive.
4. **Full LR annealing** — cosine schedule must decay to near-zero by training end. Stretched schedules (100K steps) or redistributed phase durations collapse because LR is still high late in training.
5. **Short training iterations (16)** — 32-iter training collapses at 128-256 test iters despite giving the model more "teacher forcing" signal. Fewer training iters are better for test-time scaling.
6. **No explicit fixed-point pressure** — all 4 variants (preservation weighting, L2 toward target, self-consistency, gradient masking) hurt. Contractivity emerges naturally from flat minima; explicit pressure disrupts it.

The overall picture: stability comes from landing in a flat minimum where f is naturally contractive. Gradient noise (BS), learning rate, and model capacity all control whether training finds that minimum. It's a narrow target — most hyperparameter changes break it.

## Test-Time Interventions (No Retraining)

Test-time modifications to the forward pass to see if iteration collapse can be fixed without retraining.
Scripts: `eval_interventions.py`, `modal_eval_interventions.py`.

### LR=3e-3 model (d=128, collapses at 64 iters)

| Intervention | 16 | 32 | 64 | 128 | 256 | 1024 |
|---|---|---|---|---|---|---|
| Baseline | 82.1% | 88.5% | 39.9% | 7.1% | 1.7% | 0.4% |
| Damping α=0.9 | 81.8% | 88.1% | 44.6% | 8.3% | 1.7% | 0.5% |
| Damping α=0.8 | 81.0% | 87.5% | 52.3% | 9.5% | 2.2% | 0.5% |
| Damping α=0.7 | 80.1% | 86.8% | 67.6% | 11.8% | 2.6% | 0.6% |
| Damping α=0.5 | 76.7% | 84.5% | 88.3% | 18.9% | 4.6% | 0.6% |
| Pred_scale β=0.5 | 27.2% | 31.0% | 24.1% | 7.3% | 3.4% | 0.5% |
| Pred_scale β=0.3 | 0.5% | 0.2% | 0.2% | 0.1% | 0.0% | 0.0% |
| Pred_scale β=0.1 | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Pre_norm | 81.1% | 87.7% | 18.7% | 3.7% | 0.9% | 0.1% |

### d=192 model (LR=2e-3, collapses at 128 iters)

| Intervention | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| Baseline | 84.4% | 90.8% | 94.3% | 85.9% | 23.3% | 6.9% | 3.5% |
| Damping α=0.9 | 84.2% | 90.4% | 94.1% | 87.6% | 28.3% | 6.0% | 3.8% |
| Damping α=0.8 | 83.6% | 89.9% | 93.6% | 88.7% | 34.5% | 6.0% | 3.6% |
| Damping α=0.7 | 82.7% | 89.1% | 93.0% | 90.0% | 45.3% | 7.5% | 4.0% |
| Damping α=0.5 | 79.8% | 86.8% | 91.0% | 91.3% | 70.1% | 12.5% | 4.6% |
| Pred_scale β=0.5 | 66.6% | 76.8% | 82.5% | 80.4% | 51.2% | 44.7% | 35.5% |
| Pred_scale β=0.3 | 27.6% | 31.4% | 33.2% | 31.8% | 26.2% | 24.6% | 24.0% |
| Pred_scale β=0.1 | 4.5% | 4.8% | 4.9% | 4.9% | 4.6% | 4.5% | 4.2% |
| Pre_norm | 83.6% | 89.5% | 80.2% | 30.7% | 6.1% | 0.3% | 0.0% |

### Stable model (LR=2e-3, d=128 — SOTA)

| Intervention | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| Baseline | 81.8% | 88.5% | 92.5% | 95.3% | 97.3% | 98.5% | 98.9% |
| Damping α=0.9 | 81.6% | 88.1% | 92.3% | 95.0% | 96.9% | 98.2% | 98.7% |
| Damping α=0.8 | 81.0% | 87.5% | 91.7% | 94.5% | 96.3% | 97.8% | 98.5% |
| Damping α=0.7 | 80.2% | 86.9% | 91.1% | 94.0% | 95.9% | 97.2% | 97.9% |
| Damping α=0.5 | 76.8% | 84.5% | 89.0% | 92.4% | 94.6% | 95.6% | 94.1% |
| Pred_scale β=0.5 | 30.5% | 33.5% | 34.8% | 35.9% | 36.7% | 36.8% | 36.2% |
| Pred_scale β=0.3 | 4.5% | 4.6% | 4.6% | 4.6% | 4.6% | 4.5% | 4.3% |
| Pred_scale β=0.1 | 1.2% | 0.8% | 0.8% | 0.8% | 0.8% | 0.8% | 0.6% |
| Pre_norm | 81.3% | 88.5% | 92.8% | 95.3% | 96.7% | 51.5% | 16.7% |

### Intervention Analysis

1. **Damping delays collapse proportionally but doesn't prevent it.** For LR=3e-3: α=0.5 recovers baseline-equivalent 88.3% at 64 iters (vs 39.9%), effectively shifting the collapse point by ~1 octave. For d=192: α=0.5 peaks at 91.3% at 128 (vs 85.9% baseline) and holds 70.1% at 256 (vs 23.3%). But all damped models still collapse at higher iter counts. The oscillation is merely slowed, not eliminated.

2. **Prediction scaling is destructive for LR=3e-3 but interesting for d=192.** On LR=3e-3, even β=0.5 drops accuracy from 82% to 27% at 16 iters — the feedback loop is essential. But on d=192, β=0.5 trades peak accuracy (82.5% vs 94.3% at 64) for much gentler degradation (35.5% vs 3.5% at 1024). The model becomes worse but more stable.

3. **Pre-output normalization actively introduces collapse.** Pre_norm makes all models worse — even the stable SOTA model collapses with pre_norm (96.7% at 256, then 51.5% at 512, 16.7% at 1024). On d=192: 0.0% at 1024 (vs 3.5%). LayerNorm before the output head destroys information the model uses for stable convergence.

4. **Damping on the stable model just slows convergence.** α=0.9: 98.7% at 1024 (vs 98.9% baseline). α=0.5: peaks at 95.6% at 512 then drops to 94.1% at 1024 — heavy damping makes even the stable model degrade. Confirms damping is pure friction, not a mechanism fix.

5. **Test-time interventions can't fix a non-contractive iteration map.** The collapse is baked into the trained weights. Damping just adds friction; pred_scale breaks the mechanism; pre_norm removes information. None address the root cause.

## Jacobian Spectral Radius Analysis

Estimated the spectral radius (dominant eigenvalue magnitude) of the Jacobian df/dh at various operating points using power iteration with finite-difference JVP (100 power iterations, 50 puzzles, eps=1e-3). Script: `eval_spectral_radius.py`.

| Model | SR@16 | SR@32 | SR@64 | SR@128 | SR@256 | SR Trend |
|---|---|---|---|---|---|---|
| LR=2e-3 (stable) | 55.8 | 40.6 | 29.0 | 21.0 | 14.0 | Decreasing |
| LR=3e-3 (collapse@64) | 73.0 | 64.0 | 59.9 | 63.0 | 65.1 | Flat/increasing |
| LR=1e-3 (stagnation) | 35.5 | 28.8 | 27.8 | 26.9 | 26.2 | Flat (lowest) |
| d=192 (collapse@128) | 88.0 | 69.8 | 67.1 | 76.9 | 79.2 | Decreasing then increasing |

Key findings:
1. **ALL models have SR >> 1 everywhere** — even the stable SOTA model (SR=14-56). The standard linear stability condition (SR < 1 ⟹ stable) does not apply. Convergence is entirely nonlinear.
2. **The SR trend predicts stability, not the SR magnitude.** The stable model's SR decreases monotonically (56→14), approaching contractivity. Collapsing models' SR stays flat (LR=3e-3: 60-73) or rebounds (d=192: 67→79 past iter 64). The monotonic decrease in SR is the signature of approaching a nonlinear basin of attraction.
3. **LR=1e-3 (stagnation) has the lowest SR** (26-36) — more "locally contractive" than the stable model, yet performs worse. It converges too aggressively to a suboptimal fixed point. The stable model's higher SR at early iterations means it explores more before settling.
4. **d=192's SR rebounds at the collapse point.** SR decreases from 88→67 (iters 16-64) then increases to 79 at 128 — exactly where accuracy collapses. The iteration map becomes less contractive at the point where the model needs stability most.

**Implication:** The iteration dynamics are fundamentally nonlinear. The model doesn't converge because the Jacobian has eigenvalues < 1 — it converges because the nonlinear trajectory enters a basin of attraction despite local instability. The "flat minimum" hypothesis from training may be about the geometry of these basins, not linear contractivity.

## Key Findings

1. **BS=2048 is the sweet spot for iteration stability** — BS=4096 collapses at 48 iters, BS=1024 collapses at 256 iters, BS=2048 never collapses even at 2048 iters. Likely due to flatter minima from gradient noise.
2. **Sampling strategy doesn't matter** — curriculum vs mixed gives near-identical results in all comparisons.
3. **32-iter training (teacher forcing) hurts iteration scaling** — both 32-iter models collapse past 128-256 iters, while 16-iter BS=2048 models scale to 2048+.
4. **Model converges to an argmax-fixed-point** — at 1024 iterations, outputs are identical across consecutive steps (24513/25000 stable from iter 1022–1026). The cold-start fixed-point test was misleading — it used a cold h_prev, not the warm hidden state from iterative refinement.
5. **Convergence is emergent** — the model converges monotonically without any explicit convergence loss. Flat minima from gradient noise (BS=2048) likely make f naturally contractive.
6. **Explicit fixed-point losses all degrade iteration scaling** — 4 variants tested, all collapse earlier than baseline. The extra loss terms push the model away from the flat minimum.
7. **LR=2e-3 is the sharp optimum for iteration scaling** — at d_model=128: LR=3e-3 collapses at 64 iters, LR=2.5e-3 collapses at 128, LR=2e-3 scales to 1024 (98.9%), LR=1.5e-3 scales to 1024 (98.1%), LR=1e-3 collapses at 128. Higher LR makes the iteration map non-contractive earlier; lower LR lacks the gradient noise for flat minima.
8. **Wider models (d=192) collapse regardless of LR** — LR=2e-3 peaks at 64 iters (94.3%, better per-iteration than d=128's 92.5%) but collapses at 128. LR=1e-3 collapses at 256, LR=1.5e-3 at 64. More capacity helps per-iteration but hurts fixed-point stability — the wider model's iteration map isn't contractive enough.
9. **3-phase curriculum works if you keep phase durations** — dropping Medium+ with original durations (40K total) is stable at 95.9%, but redistributing to maintain 50K steps collapses because the LR schedule decays slower.
10. **Smaller model (d=96) peaks early then degrades** — 87.8% at 128 iters, slowly degrades to 73.2% at 2048. Not enough capacity for clean convergence.
11. **Q-head (learned halt) failed** — loss competition degrades main task.
12. **Test-time interventions can't fix collapse** — damping delays collapse by ~1 octave but all damped models still collapse. Pre-output LayerNorm actively introduces collapse even on the stable model (96.7%→51.5% at 256→512). Pred scaling is destructive. On the stable model, damping just slows convergence (α=0.5 peaks at 95.6%@512, drops to 94.1%@1024).
13. **Jacobian spectral radius >> 1 for ALL models, including stable** — SR ranges from 14-88 across all models at all operating points. Linear stability theory (SR<1) does not apply. What differentiates stable from collapsing models is the SR *trend*: stable model's SR decreases monotonically (56→14); collapsing models stay flat or rebound. Convergence is entirely nonlinear — the model enters a basin of attraction despite local instability.
14. **SOTA: 98.9%** at 1024 test iters with LR=2e-3 (exp_baseline_lr2e3). Stable at 98.8% at 2048.
