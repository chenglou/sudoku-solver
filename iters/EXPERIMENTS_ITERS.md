# Iteration Experiments

Experiments on test-time iteration scaling, adaptive stopping, and training with more iterations.

See the main [EXPERIMENTS.md](../EXPERIMENTS.md) for full results.

## Key Files

- `eval_more_iters.py` - Test model at different iteration counts (no retraining)
- `eval_confidence_stop.py` - Confidence-based and oscillation-based adaptive stopping
- `eval_fixed_point.py` - Test if correct solution is a stable fixed point of f
- `exp_32iters.py` - Train with 32 iterations for fixed-point stability
- `exp_bs2048_baseline.py` - BS=2048 control (isolate batch size effect)
- `exp_qhead.py` - Q-head learned halt signal (16 iters, negative result)
- `exp_qhead_32.py` - Q-head learned halt signal (32 iters, negative result)

## Summary

| Experiment | Accuracy | Notes |
|---|---|---|
| Baseline (16 iters, BS=4096) | 82.5% | exp_faster_2drope |
| + 32 test iters | 88.4% | No retraining, collapses at 48+ |
| + oscillation stop | 91.1% | Causal, deployable |
| + peak confidence (oracle) | 91.5% | Requires all iters retroactively |
| BS=2048 baseline (16 iters) | 81.4% | -1.1pp from halved batch |
| 32-iter training (BS=2048) | 83.0% | +1.6pp over BS=2048 control |
| 32-iter + 192 test iters | 91.8% | Monotonic, no stopping tricks needed |
| Q-head (16 iters) | 79.4% | Loss competition hurts main task |
| Q-head (32 iters) | 78.4% | Same issue |

## Key Findings

1. **Test-time iteration scaling works** — running 2x training iterations gives +5.9pp for free
2. **Oscillation detection (91.1%)** surpasses nano-trm (87.4%) with no retraining
3. **Correct solution is NOT a fixed point** — f destroys 85.6% of cells (baseline), 52.4% (32-iter trained)
4. **Training with 32 iters teaches stability** — near-zero regressions across 96+ iterations (vs 17,664 regressions for baseline)
5. **32-iter model reaches 91.8%** at 192 test iters with no stopping tricks, eventually oscillates ~256+
6. **Q-head (learned halt) failed** — loss competition degrades main task
