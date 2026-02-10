# Iteration Experiments

Experiments on test-time iteration scaling, adaptive stopping, and confound isolation.

See the main [EXPERIMENTS.md](../EXPERIMENTS.md) for full results.

## Key Files

- `eval_more_iters.py` - Test model at different iteration counts (no retraining)
- `eval_confidence_stop.py` - Confidence-based and oscillation-based adaptive stopping
- `eval_fixed_point.py` - Test if correct solution is a stable fixed point of f
- `exp_bs2048_baseline.py` - BS=2048, 16-iter, reverse curriculum (**SOTA: 98.2%** at 2048 test iters)
- `exp_bs2048_mixed.py` - BS=2048, 16-iter, mixed sampling (isolates curriculum effect)
- `exp_bs1024_curriculum.py` - BS=1024, 16-iter, reverse curriculum (tests smaller batch)
- `exp_32iters.py` - BS=2048, 32-iter, mixed sampling
- `exp_32iters_curriculum.py` - BS=2048, 32-iter, reverse curriculum (isolates training iter effect)
- `exp_qhead.py` - Q-head learned halt signal (16 iters, negative result)
- `exp_qhead_32.py` - Q-head learned halt signal (32 iters, negative result)

## Test-Time Iteration Scaling

All models trained for 50K steps. Accuracy at various test-time iteration counts:

| Model | BS | Train iters | Sampling | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp_faster_2drope | 4096 | 16 | curriculum | 82.5% | 88.4% | 20.8% | — | — | — | — | — |
| **exp_bs2048_baseline** | **2048** | **16** | **curriculum** | **81.4%** | **88.1%** | **92.4%** | **94.9%** | **96.6%** | **97.5%** | **98.1%** | **98.2%** |
| exp_bs2048_mixed | 2048 | 16 | mixed | 81.2% | 88.3% | 92.3% | 94.9% | 96.4% | 97.0% | 97.3% | — |
| exp_bs1024_curriculum | 1024 | 16 | curriculum | 79.7% | 85.9% | 89.5% | 91.6% | 54.8% | 1.2% | 0.0% | — |
| exp_32iters | 2048 | 32 | mixed | 72.4% | 83.2% | 88.7% | — | 91.8% | 88.3% | — | — |
| exp_32iters_curriculum | 2048 | 32 | curriculum | — | 83.4% | 89.6% | 78.7% | 61.6% | 47.3% | 20.7% | — |

## Other Results

| Experiment | Accuracy | Notes |
|---|---|---|
| BS=4096 + oscillation stop | 91.1% | Causal, deployable |
| BS=4096 + peak confidence (oracle) | 91.5% | Requires all iters retroactively |
| Q-head (16 iters) | 79.4% | Loss competition hurts main task |
| Q-head (32 iters) | 78.4% | Same issue |

## Fixed-Point Test

**File:** `eval_fixed_point.py`

Feed the correct solution (as one-hot softmax predictions) into f and check if it's preserved. If f were a contraction mapping, the correct solution should be a stable fixed point.

| Model | Cells preserved | Puzzles perfectly preserved |
|---|---|---|
| BS=4096 baseline (16-iter) | 14.4% | 0/25000 |
| 32-iter mixed | 47.6% | 0/25000 |

**Finding:** The correct solution is NOT a fixed point for any model. f treats correct predictions as noise and re-solves from scratch. 32-iter training improves preservation (47.6% vs 14.4%) but still zero puzzles survive intact. This means the model has zero concept of "preserve what's already correct" — nothing in training incentivizes this.

**Still untested:** Training with an explicit fixed-point loss (e.g., penalize f for changing cells that are already correct). This could teach f to be a contraction mapping.

## Teacher Forcing (32-iter training) — Disproven

**Hypothesis:** Training with 32 iterations acts as implicit teacher forcing — easy puzzles are solved by iter ~10, giving f 20+ iterations of "correct input → stay correct" signal. This should teach f to preserve correct solutions and improve iteration stability.

**Result:** Disproven. Both 32-iter models (mixed and curriculum) collapse past 128-256 iters, while 16-iter BS=2048 models scale to 2048+ without collapsing. More training iterations actively hurts iteration scaling. The implicit "teacher forcing" signal from easy puzzles is too weak — f still learns to re-solve rather than preserve.

## Key Findings

1. **BS=2048 is the sweet spot for iteration stability** — BS=4096 collapses at 48 iters, BS=1024 collapses at 256 iters, BS=2048 never collapses even at 2048 iters. Likely due to flatter minima from gradient noise.
2. **Sampling strategy doesn't matter** — curriculum vs mixed gives near-identical results in all comparisons (81.4% vs 81.2% at 16 iters, 98.1% vs 97.3% at 1024 iters).
3. **32-iter training (teacher forcing) hurts iteration scaling** — both 32-iter models collapse past 128-256 iters, while 16-iter BS=2048 models scale to 2048+.
4. **Correct solution is NOT a fixed point** — f destroys 85.6% of cells (baseline), 52.4% (32-iter trained). Explicit fixed-point training untested.
5. **Convergence at BS=2048 is emergent** — the model converges monotonically without any explicit convergence loss. This likely comes from flat minima making f naturally contractive.
6. **Q-head (learned halt) failed** — loss competition degrades main task.
7. **SOTA: 98.2%** at 2048 test iters with BS=2048 baseline (still climbing, no collapse).
