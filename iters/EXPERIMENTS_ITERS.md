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
- `exp_wider_6h_lowlr.py` - d_model=192, LR=1e-3 (tests if lower LR fixes wider collapse)
- `exp_smaller_3h.py` - d_model=96, 3 heads (tests smaller model)
- `exp_baseline_lr2e3.py` - LR=2e-3 (**NEW SOTA: 98.9%** at 1024 test iters)
- `exp_baseline_lr1e3.py` - LR=1e-3 (collapses at 128 iters)
- `exp_3phase_40k.py` - 3-phase curriculum, 40K steps (stable but lower ceiling)
- `exp_3phase_50k.py` - 3-phase curriculum, 50K steps (collapses)
- `exp_qhead.py` - Q-head learned halt signal (16 iters, negative result)
- `exp_qhead_32.py` - Q-head learned halt signal (32 iters, negative result)
- `modal_eval.py` - Modal wrapper for running eval_more_iters on GPU

## Test-Time Iteration Scaling

Accuracy at various test-time iteration counts:

| Model | Notes | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|
| **exp_baseline_lr2e3** | **LR=2e-3 (NEW SOTA)** | **81.8%** | **88.5%** | **92.5%** | **95.3%** | **97.3%** | **98.5%** | **98.9%** | **98.8%** |
| exp_bs2048_baseline | LR=1.5e-3 (prev SOTA) | 81.4% | 88.1% | 92.4% | 94.9% | 96.6% | 97.5% | 98.1% | 98.2% |
| exp_baseline_lr1e3 | LR=1e-3 | 80.9% | 87.2% | 90.9% | 89.8% | 80.2% | 50.2% | 20.8% | 5.3% |
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

## Key Findings

1. **BS=2048 is the sweet spot for iteration stability** — BS=4096 collapses at 48 iters, BS=1024 collapses at 256 iters, BS=2048 never collapses even at 2048 iters. Likely due to flatter minima from gradient noise.
2. **Sampling strategy doesn't matter** — curriculum vs mixed gives near-identical results in all comparisons.
3. **32-iter training (teacher forcing) hurts iteration scaling** — both 32-iter models collapse past 128-256 iters, while 16-iter BS=2048 models scale to 2048+.
4. **Model converges to an argmax-fixed-point** — at 1024 iterations, outputs are identical across consecutive steps (24513/25000 stable from iter 1022–1026). The cold-start fixed-point test was misleading — it used a cold h_prev, not the warm hidden state from iterative refinement.
5. **Convergence is emergent** — the model converges monotonically without any explicit convergence loss. Flat minima from gradient noise (BS=2048) likely make f naturally contractive.
6. **Explicit fixed-point losses all degrade iteration scaling** — 4 variants tested, all collapse earlier than baseline. The extra loss terms push the model away from the flat minimum.
7. **LR is the most important hyperparameter for iteration scaling** — at d_model=128: LR=2e-3 (98.9%) > 1.5e-3 (98.1%) >> 1e-3 (collapses at 128). The relationship is non-monotonic — there's an optimal band, and too low is as bad as too high.
8. **LR sweet spot depends on model size** — d=192 collapses at both LR=1.5e-3 (at 64 iters) and LR=1e-3 (at 256 iters). Wider models aren't inherently broken but need different LR tuning.
9. **3-phase curriculum works if you keep phase durations** — dropping Medium+ with original durations (40K total) is stable at 95.9%, but redistributing to maintain 50K steps collapses because the LR schedule decays slower.
10. **Smaller model (d=96) peaks early then degrades** — 87.8% at 128 iters, slowly degrades to 73.2% at 2048. Not enough capacity for clean convergence.
11. **Q-head (learned halt) failed** — loss competition degrades main task.
12. **SOTA: 98.9%** at 1024 test iters with LR=2e-3 (exp_baseline_lr2e3). Stable at 98.8% at 2048.
