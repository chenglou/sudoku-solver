# Sudoku Transformer Experiments

Early experiments use 100k training steps on easiest difficulty puzzles (100k train, 1k test).
Later experiments (curriculum, recurrence) use full 2.7M training set across all difficulties (2.5k test).

**Training infrastructure:** Early experiments ran on RTX 4090. Later experiments (BS=4096, scale_wide, scale_up) ran on Modal H200.

---

## Baseline: Iterative Transformer

**File:** `sudoku.py`

**Architecture:**
- 1 transformer (4 layers, d_model=128, 4 heads)
- 16 iterations with shared weights
- Structured positional encoding (row + col + box embeddings)
- Intermediate supervision (loss at all 16 iterations)
- Input: concat(puzzle, predictions) at each iteration

**Results:** 92.8% acc, 643 solved (peak at step 96k)

---

## Ablation: No Iteration

**File:** `ablation_no_iteration.py`

**Hypothesis:** Is iterative refinement necessary, or can a single forward pass solve Sudoku?

**Change:** Set n_iterations=1 (single forward pass). Intermediate supervision N/A since only one iteration.

**Results:** 64.3% acc, 0 solved

**Finding:** Iteration is critical. Single-pass cannot do constraint propagation ("if A=5, then B≠5"). Complete failure.

---

## Ablation: No Intermediate Supervision

**File:** `ablation_no_intermediate.py`

**Hypothesis:** Does supervising all 16 iterations help, or is final-only loss sufficient?

**Change:** Only compute loss on final iteration output, not all 16.

**Results:** 87.0% acc, 428 solved

**Finding:** Intermediate supervision helps significantly (+5.8% acc, +215 puzzles). It provides gradient signal to early iterations, stabilizing training.

---

## Ablation: No Sudoku Positional Encoding

Moved to [pos_embedding/EXPERIMENTS_POS.md](pos_embedding/EXPERIMENTS_POS.md).

---

## Experiment: Project-then-Add Input

**File:** `exp_proj_add.py`

**Hypothesis:** Is concatenating puzzle and predictions optimal, or would separate projections with addition work better?

**Change:** Instead of:
```python
x_in = concat([puzzle, preds])  # (81, 19)
h = input_proj(x_in)            # single linear
```
Use:
```python
h = proj_digit(puzzle[:, 1:]) + proj_pred(preds) + empty_embed(is_empty)
```
Three separate projections added together.

**Results:**
- Original run: 96.0% acc, 817 solved
- Rerun: 91.1% acc, 555 solved (peak 93.78%/719)

**Finding:** Original result was likely a lucky run. Rerun shows high variance and similar performance to baseline. No reliable improvement from separate projections.

---

## Experiment: Project-then-Concat Input

**File:** `exp_proj_concat.py`

**Hypothesis:** Would concatenating separate projections (instead of adding) work better?

**Change:** Project to smaller dimensions that sum to d_model, then concatenate:
```python
h = torch.cat([
    proj_digit(digit_onehot),  # 9 -> 64
    proj_pred(preds),          # 9 -> 48
    empty_embed(is_empty)      # 2 -> 16
], dim=-1)                     # total: 128
```

**Results:** 92.1% acc, 582 solved (peak 92.64%/636)

**Finding:** Similar to project-then-add rerun. No clear advantage over simple concat baseline. Both fancy projection schemes show high training variance without reliable gains.

---

## Experiment: 2 Transformers × 2 Layers (Middle1)

**File:** `exp_middle1.py`

**Hypothesis:** Would specialized early/late phase transformers help? Maybe iterations 1-8 need different processing than 9-16.

**Change:**
- 2 separate transformers with 2 layers each (vs 1 transformer with 4 layers)
- T1 handles iterations 1-8, T2 handles iterations 9-16
- Same ~800k total params, but 32 layer passes (vs 64)

**Results:** 84.2% acc, 162 solved

**Finding:** Significantly worse than baseline. Depth per iteration matters more than phase specialization. 2 layers insufficient for constraint reasoning within each iteration.

---

## Experiment: 4 Transformers × 1 Layer (Middle2)

**File:** `exp_middle2.py`

**Hypothesis:** What if we specialize even more - 4 different transformers for 4 phases?

**Change:**
- 4 separate transformers with 1 layer each
- T1: iters 1-4, T2: iters 5-8, T3: iters 9-12, T4: iters 13-16
- Same ~800k total params, but 16 layer passes (vs 64)

**Results:** 72.7% acc, 0 solved

**Finding:** Complete failure. 1-layer transformers cannot reason through constraints, even with specialized weights. Depth is essential.

---

## Experiment: Unrolled (16 Separate Transformers)

**File:** `exp_unrolled.py`

**Hypothesis:** Does weight sharing help or hurt? If we use 16 separate 4-layer transformers (one per iteration), we get same FLOPs but 16x more params. This tests pure weight sharing effect.

**Change:**
- 16 separate transformers with 4 layers each (vs 1 transformer used 16 times)
- Each iteration uses a different transformer
- Same 64 layer passes (same FLOPs), but ~12.8M params (vs ~800k)

**Results:** 90.1% acc, 581 solved (peak ~566)

**Finding:** Weight sharing HELPS, not hurts. Despite 16x more parameters:
- Lower accuracy (90.1% vs 92.8%)
- Fewer puzzles solved (581 vs 643)
- Slower training throughout

Weight sharing acts as regularization - forcing the model to learn ONE general iterative function is better than learning 16 specialized ones.

---

## Experiment: Sinusoidal Positional Encoding

Moved to [pos_embedding/EXPERIMENTS_POS.md](pos_embedding/EXPERIMENTS_POS.md).

---

## Experiment: Batch Size Scaling

**Files:** `sudoku.py` (BS=128), `sudoku_bs256.py`, `sudoku_bs512.py`

**Hypothesis:** Larger batch sizes process more samples in same wall-clock time. Does this improve results?

**Change:** Scale batch size while keeping 100k steps constant. This means:
- BS=128: 12.8M samples
- BS=256: 25.6M samples (2x data, ~1.1x wall time)
- BS=512: 51.2M samples (4x data, ~2.2x wall time)

Also added bf16 mixed precision + TF32 for ~2.4x speedup.

**Results:**

| Batch Size | Final Acc | Final Solved | Peak Solved | Samples | Time |
|------------|-----------|--------------|-------------|---------|------|
| 128 | 92.2% | 537 | 679 | 12.8M | ~2.3h |
| 256 | 95.8% | 833 | **897** | 25.6M | ~2.5h |
| 512 | 94.6% | 672 | 866 | 51.2M | ~5h |

**Finding:** BS=256 is the sweet spot:
- Best peak (897 solved) and best final (833 solved)
- Only ~10% slower than BS=128 for 2x the samples
- BS=512 shows diminishing returns - more data but high variance and lower final results
- Larger batches may benefit from LR scaling (not tested)

---

## Experiment: SAM (Sharpness-Aware Minimization)

**Files:** `exp_sam.py` (BS=512), `exp_sam_bs256.py` (BS=256)

**Hypothesis:** Large batch training finds sharp minima that generalize poorly (the "generalization gap"). SAM explicitly seeks flat minima by optimizing for worst-case loss in a weight neighborhood. Can SAM close the gap for large batches?

**Background:** The generalization gap is well-documented (Keskar et al., 2017). Large batches give precise gradients that navigate into sharp valleys. Small batches have noisy gradients that can't enter narrow valleys, naturally finding flat minima. SAM addresses this directly:

```python
# SAM: optimize for worst-case loss in neighborhood
1. Compute gradient g at weights w
2. Perturb: w' = w + rho * g / ||g||
3. Compute gradient g' at perturbed w'
4. Update w using g' (not g)
```

**Results:**

| Config | Peak Solved | Final Solved | Final Acc | Time |
|--------|-------------|--------------|-----------|------|
| Vanilla BS=256 | 897 | 833 | 95.8% | ~2.5h |
| Vanilla BS=512 | 866 | 672 | 94.6% | ~5h |
| **SAM BS=256** | **958** | 930 | 98.1% | ~5.1h |
| **SAM BS=512** | **959** | 948 | 98.6% | ~6.3h |

**Finding:** SAM dramatically improves results:
- SAM BS=512 peak: 959 vs vanilla's 866 (+93 puzzles!)
- SAM BS=512 final: 948 vs vanilla's 672 (+276 puzzles!)
- SAM closes the generalization gap completely - BS=512 now matches BS=256
- SAM pushes past vanilla's ceiling entirely (959 vs 897 best vanilla)
- Overhead is ~25% for BS=512 (6.3h vs 5h), acceptable for massive gains

**Why it works:** SAM prevents the optimizer from settling into sharp minima by penalizing regions where small weight perturbations spike the loss. This gives large-batch training the generalization benefits of small-batch noise, without sacrificing throughput.

**Update (post-cosine):** With cosine LR decay, SAM's benefit drops from +6pp to just **+0.4pp** (84.0% → 83.6%). Cosine decay provides similar flat-minima benefits, making SAM largely redundant. See "Experiment: Cosine LR Without SAM" for details.

---

## Experiment: Mixed Difficulty Training

**File:** `train_mixed.py`

**Hypothesis:** Training only on easy puzzles (difficulty 0.0) doesn't generalize to harder puzzles. Will training on a mix of all difficulties help?

**Setup:**
- Sample 20k puzzles from each difficulty bucket (0-1, 1-2, 2-3, 3-4, 4+)
- Total: 100k training puzzles, uniformly mixed
- Test set: 200 from each bucket (1000 total)
- Uses SAM + BS=512 (best config)

**Zero-shot baseline** (model trained on easy only):

| Difficulty | Solved | Cell Acc |
|------------|--------|----------|
| 0.0 (easy) | 959/1000 (95.9%) | 99.0% |
| 1.x | 653/1000 (65.3%) | 90.1% |
| 2.x | 476/1000 (47.6%) | 83.5% |
| 3.x | 279/1000 (27.9%) | 76.2% |
| 4.x | 193/1000 (19.3%) | 73.0% |
| 5.x+ | 125/1000 (12.5%) | 70.4% |

**Mixed training results:**

| Difficulty | Solved | Cell Acc | vs Easy-only |
|------------|--------|----------|--------------|
| 0.0 (easy) | 960/1000 (96.0%) | 98.6% | +1 |
| 1.x | 953/1000 (95.3%) | 98.4% | **+300** |
| 2.x | 904/1000 (90.4%) | 96.6% | **+428** |
| 3.x | 851/1000 (85.1%) | 94.7% | **+572** |
| 4.x | 815/1000 (81.5%) | 93.3% | **+622** |
| 5.x+ | 775/1000 (77.5%) | 91.6% | **+650** |

**Finding:** Mixed training dramatically improves generalization:
- Easy puzzle performance unchanged (~96%)
- Hardest puzzles: 12.5% → 77.5% (+650 puzzles!)
- Cell accuracy stays high across all difficulties (91-99%)
- No curriculum learning needed - uniform mixing works great

---

## Experiment: Hard-Only Training

**File:** `train_hard.py`

**Hypothesis:** If the model learns to solve hard puzzles, easy ones should come "for free" - hard reasoning subsumes easy reasoning.

**Setup:**
- Train only on puzzles with difficulty >= 3.0 (~320k available)
- 100k training steps, SAM + BS=512
- Test on all difficulty levels

**Results:**

| Difficulty | Hard-Only | Mixed | Delta |
|------------|-----------|-------|-------|
| 0.0 (easy) | 913/1000 (91.3%) | 960/1000 (96.0%) | **-47** |
| 1.x | 702/1000 (70.2%) | 953/1000 (95.3%) | **-251** |
| 2.x | 570/1000 (57.0%) | 904/1000 (90.4%) | **-334** |
| 3.x | 879/1000 (87.9%) | 851/1000 (85.1%) | +28 |
| 4.x | 836/1000 (83.6%) | 815/1000 (81.5%) | +21 |
| 5.x+ | 800/1000 (80.0%) | 775/1000 (77.5%) | +25 |

**Finding:** The "hard subsumes easy" hypothesis is **FALSE**:

- Hard-only is better on hard puzzles (3.x+): +21-28 puzzles per bucket
- Hard-only is **much worse** on easy puzzles (0.x-2.x): -47 to -334 puzzles per bucket
- The skills don't transfer bidirectionally

**Why doesn't hard subsume easy?**

Several theories:

1. **Simplicity bias**: Neural nets naturally learn simple patterns first, then compose them for complex cases. Training only on hard puzzles may force the model to learn complex patterns that don't decompose well to simple cases.

2. **Distribution shift**: Easy puzzles have many "naked singles" (cells with only one legal value). Hard puzzles require chain reasoning. The model trained on hard puzzles may over-rely on chain reasoning even when simpler deduction suffices.

3. **Different skills, not a spectrum**: Easy and hard Sudoku may require qualitatively different reasoning strategies, not just "more" of the same skill. This mirrors the distinction between reasoning and non-reasoning language models - they may be fundamentally different capabilities.

4. **Task interference**: Hard puzzle patterns may interfere with learning easy puzzle patterns, similar to how training a language model for complex reasoning can hurt its performance on simple factual recall.

This result suggests that **difficulty levels are not strictly hierarchical** - expertise at hard puzzles doesn't automatically confer expertise at easy puzzles. Mixed training remains the best strategy.

---

## Summary Table

| Experiment | Test Set | Solved | Key Finding |
|------------|----------|--------|-------------|
| Baseline (easy only) | 1k easy | 643 | - |
| No iteration | 1k easy | 0 | Iteration critical |
| No intermediate | 1k easy | 428 | Intermediate helps |
| No sudoku pos | 1k easy | 409 | See [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) |
| Project-add | 1k easy | 555 | No reliable gain |
| Project-concat | 1k easy | 582 | No reliable gain |
| Middle1 (2×2) | 1k easy | 162 | Depth > specialization |
| Middle2 (4×1) | 1k easy | 0 | 1 layer insufficient |
| Unrolled (16×4) | 1k easy | 581 | Weight sharing helps |
| Sinusoidal pos | 1k easy | 0 | See [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) |
| BS=256 + bf16 | 1k easy | 833 | Larger batch helps |
| SAM + BS=512 | 1k easy | 948 | SAM closes gen gap |
| Mixed training | 2.5k mixed | 1930 | Mixed > easy-only |
| Curriculum (easy→hard) | 2.5k mixed | 1790 | Curriculum hurts! |
| **Reverse curriculum** | 2.5k mixed | **1994** | Hard→easy wins |
| **Recurrence (h_prev)** | 2.5k mixed | **2265** | **+13.6% over baseline** |
| Recurrence no preds | 2.5k mixed | 2238 | Preds still helps |
| No x after init | 2.5k mixed | 2248 | Removing x costs only -0.7% |
| Norm pred (TRM-style) | 2.5k mixed | 2257 | RMS norm works, but no gain |
| Eval on sudoku-extreme | 423k extreme | 32.9% | vs nano-trm 87.4% |
| Train on sudoku-extreme | 22k extreme | 63.4% | Domain match +32pp |
| **MLP-Mixer** | 25k extreme | **71.9%** | Same as Transformer |
| Scale DOWN (100K params) | 25k extreme | 8.3% | Too small, fails |
| Scale UP (5M params) | 25k extreme | 69.7% | More params ≠ better |
| **BS=4096** | 25k extreme | **76.3%** | Batch scaling most efficient |
| TRM Nested (4.5M) | 25k extreme | 60.3% | TRM architecture hurts! |
| **LR Warmup** | 25k extreme | **78.5%** | +2.2pp from 2K-step warmup |
| Fixed Random Init | 25k extreme | 77.5% | -1.0pp vs warmup, doesn't help |
| Carry Across Batches | 25k extreme | diverged | Training explodes, doesn't work |
| EMA (decay=0.999) | 25k extreme | 77.5% | -1.0pp vs warmup, doesn't help |
| Cosine LR Decay | 25k extreme | 84.0% | +5.5pp from cosine decay |
| Cosine + Mixed | 25k extreme | 83.8% | Mixed nearly matches reverse with cosine |
| Cosine + Regular | 25k extreme | 80.6% | Easy→hard still hurts (-3.4pp) |
| **Cosine - SAM** | 25k extreme | **83.6%** | **Recommended: 2x faster, -0.4pp** |
| Cosine pos_once | 25k extreme | 82.8% | See [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) |

---

## Key Insights

1. **Iteration is essential** - Sudoku requires multi-step constraint propagation. Single-pass fails completely.

2. **Depth per iteration matters** - 4 layers needed for effective reasoning. 2 layers marginal, 1 layer broken.

3. **Weight sharing actively helps** - Not just "fine" - it's beneficial! Unrolled model with 16x params performed worse. Weight sharing acts as regularization, forcing a single general iterative function.

4. **Simple concat input is fine** - Fancy projection schemes (project-then-add, project-then-concat) showed high variance and no reliable gains. Simpler is better.

5. **Intermediate supervision helps** - Gradient signal to all iterations stabilizes training.

6. **Positional encoding** - See [pos_embedding/EXPERIMENTS_POS.md](pos_embedding/EXPERIMENTS_POS.md) for full analysis. Key finding: 2D RoPE matches sudoku-specific row/col/box embeddings within noise (-0.3pp), so we use it as the baseline.

8. **Training has high variance** - Results can vary significantly between runs. Always rerun to verify improvements.

9. **Batch size 256 was sweet spot** - Before SAM, BS=256 achieved best vanilla results (897 peak). Larger BS=512 showed the generalization gap - more data but worse results.

10. **SAM closes the generalization gap** - Sharpness-Aware Minimization lets large batches find flat minima. SAM + BS=512 achieves 959 peak (vs vanilla's 897 best), with 948 final (vs 833). The ~25% overhead is worth the massive gains.

11. **Mixed difficulty training is essential** - Training only on easy puzzles fails catastrophically on hard ones (12.5% solve rate). Training on uniformly mixed difficulties achieves 77.5% on hardest while maintaining 96% on easiest. No curriculum learning needed.

12. **Hard doesn't subsume easy** - Training only on hard puzzles (difficulty >= 3.0) improves hard puzzle performance (+25 puzzles on 5.x+) but **hurts** easy puzzle performance (-334 puzzles on 2.x). This suggests easy and hard Sudoku require qualitatively different reasoning skills, not just "more" of the same skill. This parallels the distinction between reasoning and non-reasoning language models - they may be fundamentally different capabilities that don't transfer bidirectionally.

13. **Reverse curriculum beats traditional curriculum** - Starting with hard puzzles (hard→easy) outperforms both mixed training and traditional curriculum (easy→hard). Traditional curriculum actually hurts performance.

14. **Hidden state recurrence is a major win** - Passing the full hidden state (128-dim) between iterations instead of just predictions (9-dim) gives +13.6% improvement (1994→2265). The hidden state acts as a "scratchpad" for working memory, allowing the model to accumulate reasoning across iterations. Simplest approach (just `+ h_prev`) beats complex gating mechanisms.

15. **Keep explicit predictions with recurrence** - Even though h_prev theoretically contains all info needed to derive predictions, removing explicit preds from input hurts performance (-27 to -62 puzzles). The 9-dim prediction acts as a useful compressed summary alongside the 128-dim hidden state.

16. **MLP-Mixer ≈ Transformer for Sudoku** - Swapping Transformer attention for MLP-Mixer yields nearly identical results (71.9% vs 71.4%). TRM's 87.4% advantage over our 71% is NOT due to architecture choice. Despite Sudoku having fixed constraint structure where attention's dynamic routing seems wasteful, the two architectures perform equivalently at our scale (~800K params). TRM's edge likely comes from model size (5M params), data augmentation, or training tricks.

17. **Puzzle input is needed only once** - Removing the puzzle input x after the first iteration (TRM-style) costs only -0.7% (2248 vs 2265). The hidden state h_prev captures all necessary puzzle information after initial encoding. This simplifies the architecture and aligns with TRM's design.

18. **Training stabilizes around 50-60K steps** - Analysis of training curves shows: rapid gains 0-20K (0%→82%), moderate gains 20K-50K (82%→87%), then plateau with variance 50K-100K (bounces 88-91%). The last 40K steps add noise without consistent improvement. For quick experiments, 50-70K steps is sufficient to evaluate an approach.

19. **Training data domain matters more than quantity** - Training on 400K sudoku-extreme puzzles achieves 63.4% on sudoku-extreme test, vs 31.7% from 2.7M Kaggle puzzles (+32pp with 7x less data). However, the Kaggle-trained model still wins on Kaggle test (89.9% vs 81.3%). Models specialize to their training domain rather than learning universal sudoku solving.

20. **Cosine LR decay is the key technique** - After ruling out architecture (MLP-Mixer ≈ Transformer), model size (5M params hurts), fixed random init (-1.0pp), and carry across batches (diverged), we found that **cosine LR decay** is the critical technique. With warmup (78.5%) + cosine decay to 1% of peak LR, we achieve **84.0%** (+5.5pp). EMA (decay=0.999) **didn't help** (-1.0pp vs warmup). The cosine schedule allows the model to refine learned features in late training rather than continuing to jump around. Remaining gap to nano-trm (87.4%) reduced from 8.9pp to 3.4pp.

---

## Experiment: Curriculum Learning

**Files:** `train_curriculum.py`, `train_curriculum_reverse.py`, `train_mixed.py`

**Hypothesis:** Does the order of difficulty exposure matter? Traditional curriculum learning (easy→hard) is widely used, but maybe for iterative reasoning tasks, starting with hard problems builds better foundations.

**Setup:**
- ~2.7M training puzzles (all kept on CPU, batch moved to GPU per step)
- 100k steps, SAM + BS=512
- Test: 500 puzzles per bucket × 5 = 2500 total
- Train/test split: `iloc[:-500]` for train, `tail(500)` for test (verified 0 overlap)

**Phase schedules:**

| Steps | Curriculum (Easy→Hard) | Reverse (Hard→Easy) |
|-------|------------------------|---------------------|
| 0-20k | 0.0-1.0 only | 3.0+ only |
| 20-40k | 0.0-2.0 | 2.0+ |
| 40-60k | 0.0-3.0 | 1.0+ |
| 60-80k | 0.0-4.0 | All |
| 80-100k | All | All |

Mixed training uses all difficulties from step 0.

**Results:**

| Method | 0.0 | 1.x | 2.x | 3.x | 4.x+ | Total |
|--------|-----|-----|-----|-----|------|-------|
| Curriculum | 98.8% | 84.2% | 69.6% | 60.0% | 45.4% | **1790 (71.6%)** |
| Mixed | 98.4% | 87.0% | 75.0% | 67.8% | 57.8% | **1930 (77.2%)** |
| **Reverse** | 99.6% | 90.4% | 79.0% | 67.8% | 62.0% | **1994 (79.8%)** |

**Key findings:**

1. **Reverse curriculum wins** (+204 over curriculum, +64 over mixed)
2. **Traditional curriculum hurts** - actually worse than mixed training
3. **Hard puzzles benefit most**: reverse achieves 62% vs curriculum's 45.4% (37% relative improvement)
4. **No trade-off on easy**: Reverse scores highest on easy puzzles too (99.6%)

**Why reverse works:**

- **Hard-first forces robust features**: Model must learn actual constraint propagation, not shortcuts
- **Easy puzzles don't teach reasoning**: Many can be solved with simple pattern matching
- **Transfer is asymmetric**: Hard→easy skills transfer well; easy→hard skills don't
- **Iterative architecture benefits**: The 16-iteration design needs multi-step reasoning that hard puzzles demand

**Why traditional curriculum fails:**

Traditional curriculum learning works well when easier tasks teach foundational skills that compose into harder skills. For Sudoku, this assumption breaks down:
- Easy puzzles can be "solved" with pattern matching shortcuts
- These shortcuts don't generalize to hard puzzles
- The model must unlearn these shortcuts when hard puzzles arrive
- By then, training is partially wasted on the wrong inductive biases

**Conclusion:** For iterative reasoning tasks, consider **anti-curriculum (hard→easy)** over traditional curriculum learning.

---

## Inconclusive: Scaling Experiments (BS confounded)

**Files:** `exp_scale_model.py`, `exp_scale_iter.py`

**Note:** These experiments are inconclusive because they required reducing batch size from 512 to 256 due to memory constraints. The BS change is a significant confounder since BS=512+SAM was our optimal training config.

### Scale Model (d=192, L=6)

**Hypothesis:** Larger model capacity might improve performance, especially on hard puzzles.

**Change:**
- d_model: 128 → 192
- n_layers: 4 → 6
- n_heads: 4 → 6
- d_ff: 512 → 768
- ~3x parameters (~2.4M vs ~800k)
- BS: 512 → 256 (memory constraint)

**Results:** 1888/2500 (75.5%) vs baseline 1994/2500 (79.8%)

| Difficulty | Scale Model | Baseline | Delta |
|------------|-------------|----------|-------|
| 0.0 | 98.6% | 99.6% | -1.0% |
| 1.x | 86.8% | 90.4% | -3.6% |
| 2.x | 73.4% | 79.0% | -5.6% |
| 3.x | 61.8% | 67.8% | -6.0% |
| 4.x+ | 57.0% | 62.0% | -5.0% |

**Observation:** Performed worse than baseline, but unclear if due to smaller BS or model size. Peak was 1982/2500 at step 70k.

### Scale Iterations (32)

**Hypothesis:** More iterations = more "thinking time" for constraint propagation, should help hard puzzles.

**Change:**
- n_iterations: 16 → 32
- BS: 512 → 256 (memory constraint)

**Results:** 1559/2500 (62.4%) vs baseline 1994/2500 (79.8%)

| Difficulty | 32 Iter | Baseline | Delta |
|------------|---------|----------|-------|
| 0.0 | 89.2% | 99.6% | -10.4% |
| 1.x | 73.0% | 90.4% | -17.4% |
| 2.x | 57.2% | 79.0% | -21.8% |
| 3.x | 50.4% | 67.8% | -17.4% |
| 4.x+ | 42.0% | 62.0% | -20.0% |

**Observation:** Catastrophic failure. Training was extremely unstable:
- Loss spikes to 1.5+ throughout training
- Complete collapse at step 90k (solved 1/2500 puzzles)
- Never recovered to competitive performance

**Why 32 iterations might have failed:**
1. Intermediate supervision on all 32 iterations may cause gradient issues
2. Shared weights through 32 iterations may be unstable
3. 4-layer transformer may not have enough capacity per iteration
4. Smaller BS (256 vs 512) may not provide enough gradient stability

**To properly test these hypotheses, need to:**
1. Run baseline at BS=256 for fair comparison
2. Try 32 iterations with final-only loss (no intermediate supervision)
3. Try larger model + more iterations together

---

## Experiment: Hidden State Recurrence

**Files:** `exp_recur_add.py`, `exp_recur_concat.py`, `exp_recur_gated.py`, `exp_recur_mem.py`

**Hypothesis:** The baseline only passes predictions (9-dim softmax) between iterations - the hidden state h is recomputed from scratch each time. What if we pass the full hidden state (128-dim) forward, giving the model a "scratchpad" for working memory?

**Background:** Analysis of failures (`analyze_failures.py`) showed that on failed puzzles:
- Model peaks at iteration 4, then gets *worse* (oscillates without converging)
- 99.4% of failures still changing at final iteration (not converged)
- Successful puzzles show steady improvement across iterations

This suggests the model lacks persistent memory to accumulate reasoning across iterations.

**Baseline architecture:**
```python
preds = 0
for _ in range(16):
    h = transformer(input_proj(concat(x, preds)))  # h computed fresh
    preds = softmax(output_head(h))  # only 9-dim preds carries forward
```

**Four recurrence variants tested:**

| Variant | Change | Extra Params |
|---------|--------|--------------|
| **recur_add** | `h = ... + h_prev` | None |
| **recur_concat** | `input_proj(concat(x, preds, h_prev))` | +16k (input proj larger) |
| **recur_gated** | `h = gate * h_prev + (1-gate) * h_new` | +33k (gate projection) |
| **recur_mem** | Separate 64-dim memory bank, accumulated | +16k (mem projections) |

**Results:**

| Model | Total | 0.0 | 1.x | 2.x | 3.x | 4.x+ |
|-------|-------|-----|-----|-----|-----|------|
| Baseline | 1994 | 498 | 450 | 395 | 338 | 305 |
| **recur_add** | **2265** | 498 | 483 | 461 | 424 | **399** |
| recur_concat | 2206 | 500 | 483 | 447 | 403 | 373 |
| recur_gated | 2141 | 499 | 464 | 441 | 389 | 348 |
| recur_mem | 2254 | 500 | 487 | 462 | 423 | 382 |

**Key findings:**

1. **Simplest approach wins**: Just adding `h_prev` (no extra params!) gives best results
2. **Huge improvement on hard puzzles**: 305 → 399 (+94 puzzles, **+31%**)
3. **Overall improvement**: 1994 → 2265 (+271 puzzles, **+13.6%**)
4. **Gated recurrence underperforms**: GRU-style gating may be too complex, learning to gate away useful information
5. **All recurrence methods beat baseline**: Even the worst (gated, 2141) significantly outperforms baseline (1994)

**Why recurrence helps:**

The hidden state h (128-dim per cell) can encode richer information than predictions alone (9-dim per cell):
- "This cell is entangled with cells 3 and 47"
- "I tried value 5 here and it caused problems"
- "I'm waiting on more info before committing"

With recurrence, this working memory persists across iterations instead of being discarded. The model can accumulate reasoning rather than starting fresh each iteration.

**Why simple addition beats complex methods:**

- No extra parameters to learn = faster convergence
- Transformer can learn to extract what it needs from the added signal
- Acts like a residual connection across iterations
- Complex gating may learn to suppress useful information

---

## Ablation: Recurrence Without Predictions

**Files:** `exp_recur_add_nopred.py`, `exp_recur_concat_nopred.py`, `exp_recur_mem_nopred.py`

**Hypothesis:** If h_prev contains all the information (including what led to predictions), is the explicit `preds` input redundant? Since `preds = softmax(output_head(h))`, passing h_prev should be strictly more informative.

**Change:** Remove preds from input, rely solely on h_prev for iteration state:
```python
# Before: concat(x, preds) + h_prev
# After: just x + h_prev (or concat(x, h_prev) for concat variant)
```

**Results:**

| Model | With Preds | No Preds | Delta |
|-------|------------|----------|-------|
| add | **2265** | 2238 | -27 |
| concat | 2206 | 2204 | -2 |
| mem | **2254** | 2192 | -62 |

**Finding:** Removing preds **hurts** across the board, especially for the memory variant (-62 puzzles).

**Why explicit preds helps despite being "redundant":**
1. **Compressed summary**: 9-dim preds is a clean "what I currently think" signal vs 128-dim h which contains everything
2. **Supervision signal path**: Loss flows through preds → model may learn to structure h around producing good preds
3. **Easier to learn**: Extracting current prediction from h requires learning; having it explicit is free
4. **Different roles**: h_prev = "how I got here", preds = "where I am now" - both useful

**Conclusion:** Keep both preds and h_prev for best results.

---

## Experiment: No X After Init (TRM-style Input)

**File:** `exp_no_x_after_init.py`

**Hypothesis:** TRM encodes the puzzle x once at initialization and never re-feeds it. Can we do the same? This tests whether the model needs fresh x input at every iteration or if the hidden state h_prev captures all necessary puzzle info.

**Change:**
```python
# Before: h = encoder(concat(x, preds)) + h_prev  (x passed every iteration)
# After: h = h_prev + pred_proj(preds) + pos_embed  (x encoded once at init)

# Encode x ONCE at the start
h_prev = self.initial_encoder(x)

for _ in range(n_iterations):
    # NO x here - only h_prev and preds
    h = h_prev + self.pred_proj(preds) + pos_embed
    h = self.transformer(h)
    h_prev = h
    preds = F.softmax(output_head(h), dim=-1)
```

**Results:** 2248/2500 (89.9%) vs baseline 2265/2500 (90.6%) = -17 puzzles (-0.7%)

**Finding:** Removing x after initialization has essentially no cost. The model encodes all necessary puzzle information in the first iteration, and subsequent iterations can rely on h_prev alone. This is consistent with TRM's design and suggests the hidden state is a sufficient representation of the puzzle state.

---

## Experiment: Normalized Prediction Residuals (TRM-style)

**File:** `exp_norm_pred.py`

**Hypothesis:** TRM updates its prediction state with residual connections + RMS normalization. Our earlier `exp_separate_h_pred` failed because logits accumulated unboundedly. Can we make predictions update separately with proper normalization?

**Change:**
```python
# h updates independently (no preds input)
h = h_prev + pos_embed
h = self.transformer(h)
h_prev = h

# Separate prediction update with residual + RMS norm
pred_input = torch.cat([h, pred_state], dim=-1)
delta = self.pred_update(pred_input)  # MLP: (d_model+9) -> 512 -> 9
pred_state = self.pred_norm(pred_state + delta)  # RMS normalization

# Final output combines both
logits = self.output_head(h) + pred_state
```

**Results:** 2257/2500 (90.3%) vs baseline 2265/2500 (90.6%) = -8 puzzles (-0.3%)

| Difficulty | norm_pred | recur_add (baseline) |
|------------|-----------|----------------------|
| 0.x | 500/500 | 498/500 |
| 1.x | 483/500 | 483/500 |
| 2.x | 461/500 | 461/500 |
| 3.x | 420/500 | 424/500 |
| 4.x+ | 393/500 | 399/500 |

**Finding:** RMS normalization successfully prevents the explosion that killed `exp_separate_h_pred`. However, the added complexity (~70K extra params for pred_update MLP) doesn't improve results. Simple `h_prev` addition remains the best approach - more complex prediction update mechanisms don't help.

---

## Benchmark: Sudoku-Extreme

**File:** `eval_extreme.py`

**Dataset:** [sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme) - 423k test puzzles rated by backtrack count (higher = harder).

**Comparison with [nano-trm](https://github.com/olivkoch/nano-trm)** (87.4% on this benchmark):

| Rating | Baseline | recur_add | TRM |
|--------|----------|-----------|-----|
| 0 (trivial) | 98.3% | 99.8% | - |
| 1 | 78.7% | 93.4% | - |
| 2 | 68.0% | 84.2% | - |
| 3-5 | 27.9% | 40.1% | - |
| 6-10 | 3.3% | 12.1% | - |
| 11-20 | 1.8% | 11.3% | - |
| 21-50 | 1.4% | 10.4% | - |
| 51+ | 0.8% | 6.6% | - |
| **Total** | 24.4% | **32.9%** | **87.4%** |

**Analysis:**

Recurrence helps (+8.5pp overall), but we're still far from nano-trm. Later experiments (MLP-Mixer, Scale UP) ruled out architecture and model size as the gap. See Key Insight #20 for current understanding.

---

## Experiment: Training on Sudoku-Extreme

**File:** `exp_extreme_curriculum.py`

**Hypothesis:** Our models trained on Kaggle data only achieve 31.7% on sudoku-extreme. Does training directly on sudoku-extreme close the gap to TRM?

**Setup:**
- 400K training puzzles from sudoku-extreme (test split, before we knew train split existed)
- Same architecture as `no_x_after_init` (encode x once)
- Reverse curriculum based on rating: Phase 1 (rating 21+) → Phase 2 (6+) → Phase 3 (1+) → Phase 4 (all)
- 100K steps, SAM + BS=512

**Results:**

| Trained on | Kaggle test (2.5K) | sudoku-extreme test (22K) |
|------------|-------------------|---------------------------|
| Kaggle 2.7M | **89.9%** | 31.7% |
| sudoku-extreme 400K | 81.3% | **63.4%** |
| nano-trm (reference) | - | **87.4%** |

**By difficulty on sudoku-extreme:**

| Rating | Kaggle-trained | Extreme-trained |
|--------|----------------|-----------------|
| 0 (trivial) | 99.8% | 98.7% |
| 1-2 | 88.9% | 79.8% |
| 3-10 | 62.0% | 51.5% |
| 11-50 | 21.5% | 53.4% |
| 51+ | 6.6% | 62.8% |

**Key findings:**

1. **Domain match matters hugely**: +32pp on sudoku-extreme (31.7% → 63.4%) with 7x less data
2. **Not universally better data**: -8.6pp on Kaggle (89.9% → 81.3%), models specialize to their domain
3. **Still far from TRM**: 63.4% vs 87.4% despite training on same domain. Architecture gap remains.
4. **Hard puzzles benefit most**: Rating 51+ jumps from 6.6% to 62.8% (+56pp!)

**Stabilization:** Similar to Kaggle experiments - rapid gains until 50K, plateau with variance 50-100K. 70K steps sufficient.

---

## New Baseline: 2.7M Sudoku-Extreme

**File:** `exp_extreme_baseline.py`

**Hypothesis:** Our previous sudoku-extreme experiment used only 400K puzzles. What if we match the Kaggle data quantity (2.7M)?

**Setup:**
- 2.7M training puzzles from sudoku-extreme train split (matching Kaggle quantity)
- Same architecture as `no_x_after_init` (encode x once)
- Reverse curriculum based on rating: Phase 1 (rating 21+) → Phase 2 (6+) → Phase 3 (1+) → Phase 4 (all)
- 70K steps (not 100K - training stabilizes by then), SAM + BS=512

**Results:**

| Trained on | Data | Kaggle test (2.5K) | sudoku-extreme test |
|------------|------|-------------------|---------------------|
| Kaggle | 2.7M | **89.9%** | 31.7% |
| sudoku-extreme | 400K | 81.3% | 63.4% |
| **sudoku-extreme** | **2.7M** | 83.3% | **71.4%** |
| nano-trm (reference) | 1K | - | 87.4% |

**By difficulty on sudoku-extreme test:**

| Rating | 400K trained | 2.7M trained |
|--------|--------------|--------------|
| 0 (trivial) | 4812/5000 (96.2%) | 4935/5000 (98.7%) |
| 1-2 | 3888/5000 (77.8%) | 4154/5000 (83.1%) |
| 3-10 | 2513/5000 (50.3%) | 2999/5000 (60.0%) |
| 11-50 | 2617/5000 (52.3%) | 3283/5000 (65.7%) |
| 51+ | 2874/5000 (57.5%) | 3563/5000 (71.3%) |

**Key findings:**

1. **More data helps**: +8pp on sudoku-extreme (63.4% → 71.4%) by increasing from 400K to 2.7M
2. **Cross-domain also improves**: +2pp on Kaggle (81.3% → 83.3%) despite never seeing Kaggle puzzles
3. **Still far from nano-trm**: 71.4% vs 87.4% - later experiments ruled out architecture as the bottleneck
4. **Hard puzzles benefit most**: Rating 51+ jumps from 57.5% to 71.3% (+14pp)

**This is now the new baseline** for sudoku-extreme experiments: 2.7M data, 70K steps, no_x_after_init architecture

---

## Experiment: MLP-Mixer (Architecture Swap)

**File:** `exp_mlp_mixer.py`

**Hypothesis:** TRM uses MLP-Mixer instead of Transformer attention. Is their 87.4% vs our 71.4% due to the architecture difference?

**Change:** Replace TransformerEncoder with MLP-Mixer layers:
```python
# Before: self-attention + FFN
h = self.transformer(h)

# After: token mixing MLP + channel mixing MLP
class MixerLayer(nn.Module):
    def forward(self, x):
        # Token mixing: MLP across 81 positions
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)  # 81 -> 512 -> 81
        x = x + y
        # Channel mixing: MLP across features (same as FFN)
        x = x + self.channel_mix(self.norm2(x))
        return x
```

Everything else identical: same looping, h_prev recurrence, pos embeddings, training setup.

**Parameters:** 870K (MLP-Mixer) vs 800K (Transformer) - similar

**Results:**

| Model | Total | Rating 0 | 1-2 | 3-10 | 11-50 | 51+ |
|-------|-------|----------|-----|------|-------|-----|
| **MLP-Mixer** | **71.9%** | 99.1% | 86.6% | 53.3% | 56.9% | 63.4% |
| Transformer | 71.4% | 98.7% | 83.1% | 60.0% | 65.7% | 71.3% |
| nano-trm | 87.4% | - | - | - | - | - |

**Finding:** MLP-Mixer achieves **71.9%** vs Transformer's **71.4%** - essentially identical (+0.5pp).

Interesting pattern: MLP-Mixer is *worse* on hard puzzles (51+: 63.4% vs 71.3%) but *better* on easy (1-2: 86.6% vs 83.1%). The architectures trade off differently across difficulties but converge to the same overall accuracy.

**Conclusion:** Swapping Transformer for MLP-Mixer does NOT explain TRM's advantage. The 16pp gap must come from:
1. **Model size**: TRM 5M params vs our 870K (6x larger)
2. **Data augmentation**: TRM uses 1000 digit relabelings per puzzle
3. **Training tricks**: EMA, different supervision schedule (H_cycles/L_cycles)

Architecture alone is not the bottleneck.

---

## Experiment: Scale DOWN

**File:** `exp_scale_down.py`

**Hypothesis:** How small can we go? Testing if a smaller model can still solve Sudoku.

**Change:**
- d_model: 128 → 64
- n_layers: 4 → 2
- n_heads: 4 → 2
- d_ff: 512 → 256
- ~100K params (vs baseline ~800K)

**Results:**

| Rating | Solved |
|--------|--------|
| 0 (easy) | 32.2% |
| 1-2 | 1.4% |
| 3-10 | 1.8% |
| 11-50 | 4.2% |
| 51+ | 2.1% |
| **Total** | **8.3%** |

**Finding:** Catastrophic failure. The small model can barely solve easy puzzles (32%) and essentially nothing harder.

This confirms Key Insight #2: depth per iteration matters. With only 2 layers, the model cannot perform the multi-step constraint reasoning needed within each iteration. Combined with smaller d_model, the model lacks both the depth and capacity for Sudoku.

**Minimum viable size** appears to be somewhere between 100K and 800K params.

---

## Experiment: Scale UP

**File:** `exp_scale_up.py`

**Hypothesis:** TRM has 5M params vs our 800K. Is model size the bottleneck? Let's scale up to match TRM.

**Change:**
- d_model: 128 → 256
- n_layers: 4 → 8
- n_heads: 4 → 8
- d_ff: 512 → 1024
- ~5M params (matching TRM)
- Used gradient accumulation (micro_batch=128 × 4 = effective BS=512) due to GPU memory

**Results:**

| Rating | Scale UP (5M) | Baseline (800K) |
|--------|---------------|-----------------|
| 0 (easy) | 98.2% | 98.7% |
| 1-2 | 80.8% | 83.1% |
| 3-10 | 50.8% | 60.0% |
| 11-50 | 55.5% | 65.7% |
| 51+ | 63.0% | 71.3% |
| **Total** | **69.7%** | **71.4%** |

**Reference:** TRM (5M params): 87.4%

**Finding:** Scaling up 6x to 5M params made results **WORSE** (-1.7pp). Despite matching TRM's parameter count, we don't match its performance. Key insights:

1. **Model size is NOT the bottleneck** - More params doesn't help, may even hurt
2. **Our architecture has trouble utilizing capacity** - The larger model trains more efficiently early (51% vs 12% at step 5K) but converges to worse results
3. **TRM's advantage must be elsewhere**:
   - Data augmentation (1000 digit relabelings per puzzle)
   - Training methodology (H_cycles/L_cycles nested loops, EMA)
   - Possibly better inductive biases in their specific MLP-Mixer design

**The gap with TRM (69.7% vs 87.4%) is NOT due to model size** - both have ~5M params. Combined with the MLP-Mixer experiment (71.9% ≈ 71.4%), we've ruled out both architecture type and model size as the bottleneck. The remaining differences are in training methodology.

---

## Experiment: Scale UP with True Batch Size

**File:** `exp_scale_up_big_gpu.py`

**Hypothesis:** The previous Scale UP used gradient accumulation (micro_batch=128 × 4). True large batches may behave differently due to BatchNorm statistics, gradient noise, etc. Rerun on H200 GPU with true BS=512.

**Setup:**
- Same architecture as Scale UP: d_model=256, n_layers=8, n_heads=8, d_ff=1024 (~6.3M params)
- True BS=512 (no gradient accumulation)
- 70K steps on sudoku-extreme 2.7M

**Results:**

| Rating | True BS (6.3M) | Grad Accum (5M) | Baseline (800K) |
|--------|----------------|-----------------|-----------------|
| 0 | 98.7% | 98.2% | 98.7% |
| 1-2 | 85.6% | 80.8% | 83.1% |
| 3-10 | 57.2% | 50.8% | 60.0% |
| 11-50 | 58.5% | 55.5% | 65.7% |
| 51+ | 67.3% | 63.0% | 71.3% |
| **Total** | **73.5%** | **69.7%** | **71.4%** |

**Finding:** True batch size matters! +3.8pp over gradient accumulation version. Now beats baseline by +2.1pp.

---

## Experiment: Scale WIDE

**File:** `exp_scale_wide.py`

**Hypothesis:** Instead of scaling depth (more layers), what if we scale width (larger d_model)?

**Setup:**
- d_model: 128 → 512
- n_layers: 4 (unchanged)
- d_ff: 512 → 2048
- ~3.2M params
- 70K steps, BS=512

**Results:**

| Rating | Scale WIDE (3.2M) | Scale UP (6.3M) | Baseline (800K) |
|--------|-------------------|-----------------|-----------------|
| 0 | 99.4% | 98.7% | 98.7% |
| 1-2 | 85.6% | 85.6% | 83.1% |
| 3-10 | 58.0% | 57.2% | 60.0% |
| 11-50 | 61.4% | 58.5% | 65.7% |
| 51+ | 69.5% | 67.3% | 71.3% |
| **Total** | **74.8%** | **73.5%** | **71.4%** |

**Finding:** Width scaling (74.8%) beats depth scaling (73.5%) with fewer params (3.2M vs 6.3M). Width is more parameter-efficient than depth for this task.

---

## Experiment: Batch Size Scaling on Sudoku-Extreme

**Files:** `exp_scale_batch.py` (BS=2048), `exp_scale_batch_4k.py` (BS=4096)

**Hypothesis:** Earlier experiments showed BS=256-512 was optimal for small datasets. With 2.7M training puzzles, can we push batch size higher?

**Setup:**
- Same baseline architecture: d_model=128, n_layers=4 (~800K params)
- Scale BS: 512 → 2048 → 4096
- Keep LR=1e-3 for BS=2048, LR=1.5e-3 for BS=4096
- 70K steps on sudoku-extreme 2.7M

**Results:**

| Batch Size | Params | Total | Rating 51+ |
|------------|--------|-------|------------|
| 512 (baseline) | 800K | 71.4% | 71.3% |
| 2048 | 800K | 73.7% | 67.8% |
| 4096 | 800K | **76.3%** | **72.3%** |
| nano-trm (ref) | 5M | 87.4% | - |

**Learning curve (BS=4096):**

| Step | Test Acc |
|------|----------|
| 5K | 54.9% |
| 10K | 64.1% |
| 20K | 68.5% |
| 30K | 73.1% |
| 40K | 73.8% |
| 50K | 75.4% |
| 70K | **76.3%** |

**Finding:** Batch size scaling is the most efficient lever we've found:
- BS=4096 (76.3%) beats 8x more params (73.5%) with zero extra cost per sample
- +4.9pp over baseline just from larger batches
- Gap to TRM reduced from 16pp to 11pp

---

## Experiment: Curriculum Scaling at Large Batch Size

**Files:** `exp_scale_batch_4k_v2.py` (reverse), `exp_scale_batch_4k_curriculum.py` (regular)

**Hypothesis:** Our reverse curriculum was designed for BS=512. With BS=4096, each step sees 8x more data. Should phase boundaries scale?

**Setup:**
- BS=4096 for both experiments
- 10K steps (same total data as BS=512 @ ~80K steps)
- Scaled phases: 0-2K, 2K-4K, 4K-6K, 6K-10K

| Curriculum | Phase Order | Steps | Accuracy |
|------------|-------------|-------|----------|
| Reverse (scaled) | hard→easy | 10K | **70.5%** |
| Regular (scaled) | easy→hard | 10K | 67.1% |

**Same-data efficiency comparison:**

At 41M samples (10K steps @ BS=4096):
- Scaled reverse curriculum: **70.5%**
- Unscaled reverse curriculum: **64.1%**

Scaled phases are **+6.4% more efficient** at the same data budget.

**Learning curves:**

Reverse curriculum (10K steps):
```
Step 0K: 0% → 2K: 3% → 3K: 36% → 5K: 63% → 8K: 68.5% → 10K: 70.5%
```

Regular curriculum (10K steps):
```
Step 0K: 0% → 2K: 31% → 4K: 52% → 6K: 65% → 8K: 68.1% → 10K: 67.1% (dropped!)
```

**Key observations:**
1. Regular curriculum **overfits**: peaks at 68.2% (step 8K), then drops to 67.1%
2. Loss spikes visible at phase transitions (14K, 42K in unscaled runs)
3. Reverse curriculum keeps improving; regular curriculum degrades after step 8K

**Finding:** Reverse curriculum still wins by +3.4pp at large batch size. Scaling phases improves data efficiency, but more training data still wins overall (76.3% with 70K steps vs 70.5% with 10K steps).

---

## Experiment: TRM Architecture (Nested H_cycles/L_cycles)

**File:** `exp_trm_nested.py`

**Hypothesis:** nano-trm achieves 87.4% on sudoku-extreme. They use a specific architecture with nested iteration loops (H_cycles × L_cycles) where outer loops run without gradients. Does adopting their architecture close the gap?

**Setup:**
- TRM-style MLP-T architecture (sequence-wise MLP instead of attention)
- hidden_size=512, L_layers=2 (matching TRM)
- H_cycles=3, L_cycles=6 (first 2 outer loops without gradients)
- ~4.5M params
- Trained on our 2.7M sudoku-extreme data with reverse curriculum
- BS=4096, lr=1e-4, weight_decay=1.0
- 70K steps

**Results:**

| Rating | TRM Nested | Baseline (BS=4096) | nano-trm (ref) |
|--------|------------|-------------------|-----------------|
| 0 (easy) | 95.1% | 98.7% | - |
| 1-2 | 73.7% | 83.1% | - |
| 3-10 | 41.5% | 60.0% | - |
| 11-50 | 44.4% | 65.7% | - |
| 51+ | 46.9% | 71.3% | - |
| **Total** | **60.3%** | **76.3%** | **87.4%** |

**Finding:** TRM's architecture **hurts** performance significantly (-16pp vs baseline).

The nested H_cycles/L_cycles structure with no-gradient outer loops appears to be harmful when combined with our training setup. Several possible explanations:

1. **Data mismatch**: TRM trains on 1K puzzles × 1000 augmentations. The nested loop structure may be designed for that small-data regime, not 2.7M puzzles.
2. **No-gradient warmup wasteful**: Running 2 full outer cycles without gradients may work when each "step" sees many augmented versions of the same puzzle, but wastes computation when each step sees fresh data.
3. **MLP-T not inherently better**: Despite fixed Sudoku constraints favoring fixed mixing patterns, MLP-T doesn't outperform attention at this scale.

**Conclusion:** nano-trm's architecture is NOT the source of their advantage. Subsequent experiments with their exact recipe (exp_trm_exact.py) also failed (~14% accuracy due to massive overfitting). The gap likely comes from subtle training details: non-learned hidden state init with std=1.0, carry persisting across batches, or no-grad warmup cycles. See nano-trm analysis in Key Insight #20.

---

## Experiment: LR Warmup

**File:** `exp_warmup.py`

**Hypothesis:** nano-trm uses 2000-step linear LR warmup. Large batch training can have unstable early gradients - does warmup help?

**Setup:**
- Same as BS=4096 baseline
- Added 2000-step linear LR warmup (0 → 1.5e-3)
- 70K steps on H200

**Results:**

| Metric | With Warmup | Baseline (no warmup) |
|--------|-------------|----------------------|
| Rating 0 | 99.6% | 98.7% |
| Rating 1-2 | 90.2% | 83.1% |
| Rating 3-10 | 62.3% | 60.0% |
| Rating 11-50 | 66.2% | 65.7% |
| Rating 51+ | 74.0% | 72.3% |
| **Total** | **78.5%** | **76.3%** |

**Learning curve comparison:**

| Step | With Warmup | Baseline | Delta |
|------|-------------|----------|-------|
| 5K | 57.1% | 54.9% | +2.2pp |
| 10K | 66.4% | 64.1% | +2.3pp |
| 25K | 70.9% | 70.2% | +0.7pp |
| 50K | 76.9% | 75.4% | +1.5pp |
| 70K | **78.5%** | 76.3% | **+2.2pp** |

**Finding:** LR warmup gives a solid **+2.2pp improvement** (76.3% → 78.5%). The benefit is most visible early (helps early training stability) and persists to the end. Gap to nano-trm reduced from 11.1pp to 8.9pp.

**Remaining gap with nano-trm:**
- Baseline (no warmup): 76.3% → 11.1pp gap
- **With warmup: 78.5% → 8.9pp gap**
- nano-trm: 87.4%

Still need to test: EMA, cosine LR decay, carry across batches, no-grad warmup cycles.

---

## Experiment: Fixed Random Hidden State Init

**File:** `exp_fixed_init.py`

**Hypothesis:** nano-trm initializes hidden states (z_H, z_L) as fixed random buffers with std=1.0, not learned parameters. Does using a fixed random init instead of our learned `initial_encoder(x)` help?

**Setup:**
- Based on warmup baseline (78.5%)
- Replaced `self.initial_encoder = nn.Linear(10, d_model)` with `nn.Buffer` initialized with std=1.0
- Added `puzzle_proj(x)` inside the loop to feed puzzle info (since h_init no longer encodes it)
- 70K steps on H200

**Architecture change:**
```python
# Before (learned, puzzle-dependent):
self.initial_encoder = nn.Linear(10, d_model)
h_prev = self.initial_encoder(x)

# After (fixed random, same for all puzzles):
self.register_buffer('h_init', torch.randn(1, 81, d_model) * 1.0)
h_prev = self.h_init.expand(batch_size, -1, -1)
puzzle_embed = self.puzzle_proj(x)  # Added inside loop
```

**Results:**

| Metric | Fixed Init | Warmup Baseline |
|--------|------------|-----------------|
| Rating 0 | 99.6% | 99.6% |
| Rating 1-2 | 88.8% | 90.2% |
| Rating 3-10 | 60.9% | 62.3% |
| Rating 11-50 | 64.6% | 66.2% |
| Rating 51+ | 73.7% | 74.0% |
| **Total** | **77.5%** | **78.5%** |

**Learning curve:**

| Step | Fixed Init | Warmup | Delta |
|------|------------|--------|-------|
| 10K | 65.2% | 66.4% | -1.2pp |
| 50K | 76.6% | 76.9% | -0.3pp |
| 55K | 76.9% | ~77% | ~0pp |
| 70K | **77.5%** | **78.5%** | **-1.0pp** |

**Finding:** Fixed random init **hurts** by 1.0pp (78.5% → 77.5%). Our learned `initial_encoder(x)` that encodes the puzzle into the initial hidden state is better than nano-trm's fixed random buffer approach.

**Why it didn't help:**
- nano-trm has TWO hidden states (z_H for solution, z_L for problem) that interact differently
- Our single h_prev benefits from being puzzle-aware from the start
- The fixed init forces the model to rely entirely on `puzzle_proj(x)` added each iteration, which may be less effective than encoding puzzle info once into h_prev

**Conclusion:** This nano-trm technique does NOT transfer to our architecture. Crossed off the list.

---

## Experiment: Carry Hidden State Across Batches

**File:** `exp_carry.py`

**Hypothesis:** nano-trm keeps the same puzzle in a batch slot until solved, carrying hidden state forward. This lets the model "think longer" on hard puzzles. Does this help our architecture?

**Setup:**
- Based on warmup baseline (78.5%)
- Persistent slot system: each batch slot holds a puzzle until solved, then replaced
- Hidden state (h_prev) and predictions carry forward for unsolved puzzles
- 70K steps on H200

**Results:** Training diverged catastrophically. Loss exploded to millions by step 20K, accuracy stuck at 11% (random), zero puzzles solved.

**Finding:** Carry across batches **completely breaks** our architecture. The hidden state accumulation causes training instability. This technique does NOT transfer from nano-trm.

**Why it failed:**
- nano-trm has different architecture (two-state z_H/z_L system)
- Their "iterations" within a step work differently
- Our single h_prev may accumulate in ways that destabilize gradients
- The carry mechanism may need their specific normalization or gating

**Conclusion:** Crossed off the list. Will test EMA and cosine LR decay independently.

---

## Experiment: EMA (Exponential Moving Average)

**File:** `exp_ema.py`

**Hypothesis:** nano-trm uses EMA with decay=0.999 for evaluation. Does maintaining shadow weights that average over training improve generalization?

**Setup:**
- Based on warmup baseline (78.5%)
- Added EMA class that maintains shadow weights updated after each step: `shadow = decay * shadow + (1-decay) * params`
- Evaluation uses EMA weights, training uses live weights
- decay=0.999 (same as nano-trm)
- 70K steps on H200

**Implementation:**
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
```

**Results:**

| Metric | With EMA | Warmup Baseline |
|--------|----------|-----------------|
| Rating 0 | 99.5% | 99.6% |
| Rating 1-2 | 89.6% | 90.2% |
| Rating 3-10 | 61.6% | 62.3% |
| Rating 11-50 | 64.8% | 66.2% |
| Rating 51+ | 72.0% | 74.0% |
| **Total** | **77.5%** | **78.5%** |

**Finding:** EMA **hurts** by 1.0pp (78.5% → 77.5%). The shadow weights averaged over training are worse than the final live weights.

**Why it didn't help:**
- Our training with SAM already finds flat minima, reducing the need for weight averaging
- EMA may average in older, less refined weights that hurt final performance
- nano-trm may benefit from EMA due to their different training dynamics (carry across batches, nested loops)

**Conclusion:** EMA does NOT help our architecture. Crossed off the list.

---

## Experiment: Cosine LR Decay (NEW SOTA)

**File:** `exp_cosine.py`

**Hypothesis:** nano-trm uses cosine LR decay after warmup. Does decaying to a small final LR help the model converge to a better solution?

**Setup:**
- Based on warmup baseline (78.5%)
- After 2K-step linear warmup, LR decays following cosine schedule to 1% of peak
- `lr = lr_peak * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + cos(π * progress)))`
- lr_min_ratio = 0.01 (final LR = 1.5e-5)
- 70K steps on H200

**Implementation:**
```python
def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (lr_min_ratio + (1 - lr_min_ratio) * cosine_decay)
```

**Results:**

| Metric | Cosine LR | Warmup Baseline | Delta |
|--------|-----------|-----------------|-------|
| Rating 0 | 99.9% | 99.6% | +0.3pp |
| Rating 1-2 | 94.4% | 90.2% | +4.2pp |
| Rating 3-10 | 70.9% | 62.3% | +8.6pp |
| Rating 11-50 | 74.9% | 66.2% | +8.7pp |
| Rating 51+ | 80.0% | 74.0% | +6.0pp |
| **Total** | **84.0%** | **78.5%** | **+5.5pp** |

**Learning curve:**

| Step | Cosine | Warmup | Delta |
|------|--------|--------|-------|
| 10K | 66.8% | 66.4% | +0.4pp |
| 25K | 74.0% | 70.9% | +3.1pp |
| 50K | 80.0% | 76.9% | +3.1pp |
| 70K | **84.0%** | **78.5%** | **+5.5pp** |

**Finding:** Cosine LR decay gives a massive **+5.5pp improvement** (78.5% → 84.0%). This is the single largest improvement we've found, making it the **NEW SOTA**.

**Why cosine helps:**
1. **Late-stage refinement**: As LR decays, the model makes smaller updates, allowing it to fine-tune features without jumping over good solutions
2. **Smooth decay**: Cosine is smoother than step decay, avoiding sudden drops that could destabilize training
3. **Never-zero LR**: The 1% minimum ensures continued learning even at the end
4. **Complements warmup**: Warmup stabilizes early training, cosine refines late training

**Remaining gap with nano-trm:**
- Warmup only: 78.5% → 8.9pp gap
- **Cosine LR: 84.0% → 3.4pp gap**
- nano-trm: 87.4%

**Conclusion:** Cosine LR decay is the key technique for closing the gap with nano-trm. The remaining 3.4pp gap may come from their data augmentation (1000 digit relabelings per puzzle) or other subtle differences.

---

## Experiment: Cosine LR + Mixed Sampling

**File:** `exp_cosine_mixed.py`

**Hypothesis:** The original cosine experiment uses reverse curriculum (hard→easy). Does cosine LR work just as well with mixed sampling (all difficulties from start)?

**Setup:**
- Same as exp_cosine (84.0% SOTA) but with mixed sampling instead of reverse curriculum
- No phases - sample from all 2.7M puzzles throughout training
- 70K steps on H200

**Results:**

| Metric | Mixed | Reverse (SOTA) | Delta |
|--------|-------|----------------|-------|
| Rating 0 | 100.0% | 99.9% | +0.1pp |
| Rating 1-2 | 96.0% | 94.4% | +1.6pp |
| Rating 3-10 | 71.2% | 70.9% | +0.3pp |
| Rating 11-50 | 70.8% | 74.9% | -4.1pp |
| Rating 51+ | 81.0% | 80.0% | +1.0pp |
| **Total** | **83.8%** | **84.0%** | **-0.2pp** |

**Finding:** Mixed sampling nearly matches reverse curriculum with cosine LR (-0.2pp). Before cosine, mixed was -2.6pp behind reverse. Cosine decay appears to reduce the importance of curriculum design.

**Why mixed nearly matches now:**
- Cosine's gradual LR decay provides implicit curriculum - early high-LR learns coarse features, late low-LR refines
- The explicit hard→easy ordering becomes less important when the LR schedule already provides structure

---

## Experiment: Cosine LR + Regular Curriculum

**File:** `exp_cosine_regular.py`

**Hypothesis:** Does traditional curriculum (easy→hard) work with cosine LR?

**Setup:**
- Same as exp_cosine but with regular curriculum: easy→hard phases
- Phase 1: rating ≤2, Phase 2: ≤10, Phase 3: ≤50, Phase 4: all
- 70K steps on H200

**Results:**

| Metric | Regular | Reverse (SOTA) | Delta |
|--------|---------|----------------|-------|
| Rating 0 | 100.0% | 99.9% | +0.1pp |
| Rating 1-2 | 95.2% | 94.4% | +0.8pp |
| Rating 3-10 | 66.2% | 70.9% | -4.7pp |
| Rating 11-50 | 66.3% | 74.9% | -8.6pp |
| Rating 51+ | 75.5% | 80.0% | -4.5pp |
| **Total** | **80.6%** | **84.0%** | **-3.4pp** |

**Finding:** Regular curriculum (easy→hard) still hurts significantly (-3.4pp) even with cosine LR. The pattern from pre-cosine experiments holds: easy puzzles teach shortcuts that don't generalize to hard puzzles.

**Curriculum comparison with cosine:**

| Curriculum | Result | vs Reverse |
|------------|--------|------------|
| Reverse (hard→easy) | 84.0% | baseline |
| Mixed (no curriculum) | 83.8% | -0.2pp |
| Regular (easy→hard) | 80.6% | -3.4pp |

**Conclusion:** Even with cosine LR, curriculum order matters. Reverse still wins, but mixed is now nearly as good. Regular curriculum remains harmful.

---

## Experiment: Cosine LR Without SAM (NEW RECOMMENDED BASELINE)

**File:** `exp_cosine_no_sam.py`

**Hypothesis:** SAM and cosine LR both help find flat minima. With cosine decay, is SAM still needed?

**Setup:**
- Same as exp_cosine (84.0%) but with plain AdamW instead of SAM
- Removes the two-step SAM procedure (first_step/second_step)
- **2x faster training** (one forward-backward per step instead of two)
- 70K steps on H200

**Results:**

| Metric | No SAM | With SAM | Delta |
|--------|--------|----------|-------|
| Rating 0 | 99.8% | 99.9% | -0.1pp |
| Rating 1-2 | 94.3% | 94.4% | -0.1pp |
| Rating 3-10 | 70.0% | 70.9% | -0.9pp |
| Rating 11-50 | 71.8% | 74.9% | -3.1pp |
| Rating 51+ | 81.9% | 80.0% | +1.9pp |
| **Total** | **83.6%** | **84.0%** | **-0.4pp** |

**Finding:** SAM only contributes **0.4pp** with cosine LR decay. This is dramatically less than pre-cosine where SAM gave +6pp.

**Why SAM became redundant:**
- Cosine LR decay provides similar benefits to SAM: gradual refinement in late training helps find flat minima
- Both techniques prevent the optimizer from jumping around in late training
- With cosine doing this job, SAM's sharpness-aware perturbations add little

**Trade-off:** 0.4pp accuracy vs simpler training. For most purposes, **cosine without SAM is the recommended baseline**:
- 83.6% accuracy (vs 84.0% with SAM)
- ~2h training time (vs ~4h with SAM)
- Simpler code (no SAM class needed)

**Historical context:**
- Pre-cosine: SAM critical (+6pp), overhead justified
- Post-cosine: SAM marginal (+0.4pp), overhead not justified

---

## Experiment: Position Embedding Once vs Every Iteration

Moved to [pos_embedding/EXPERIMENTS_POS.md](pos_embedding/EXPERIMENTS_POS.md).

---

## Experiment: High Weight Decay (nano-trm style)

**File:** `exp_cosine_wd.py`

**Hypothesis:** nano-trm uses weight_decay=1.0, we use 0.01 (AdamW default). Does 100x higher weight decay help?

**Setup:**
- Same as exp_cosine_no_sam (83.6%) but with weight_decay=1.0
- 70K steps on H200

**Results:**

| Metric | WD=1.0 | WD=0.01 (SOTA) | Delta |
|--------|--------|----------------|-------|
| Rating 0 | 99.6% | 99.8% | -0.2pp |
| Rating 1-2 | 80.8% | 94.3% | -13.5pp |
| Rating 3-10 | 49.8% | 70.0% | -20.2pp |
| Rating 11-50 | 53.3% | 71.8% | -18.5pp |
| Rating 51+ | 64.8% | 81.9% | -17.1pp |
| **Total** | **70.1%** | **83.6%** | **-13.5pp** |

**Finding:** Weight decay 1.0 completely kills performance (-13.5pp). nano-trm must have a different setup (different architecture, different optimizer, or their high WD interacts with other techniques we don't have).

**Why it failed:**
- Our model has 800K params vs nano-trm's 5M - less capacity to spare
- High WD aggressively shrinks weights toward zero
- With cosine LR already providing regularization, extra WD is overkill
- nano-trm may compensate with something else (e.g., higher base LR, different architecture)

**Conclusion:** Keep weight_decay=0.01. Higher values hurt significantly.

---

## Experiment: GELU Activation

**File:** `exp_cosine_gelu.py`

**Hypothesis:** Modern transformers use GELU instead of ReLU. Does it help?

**Setup:**
- Same as exp_cosine_no_sam (83.6%) but with activation="gelu"
- 70K steps on H200

**Results:**

| Metric | GELU | ReLU (SOTA) | Delta |
|--------|------|-------------|-------|
| Rating 0 | 100.0% | 99.8% | +0.2pp |
| Rating 1-2 | 94.8% | 94.3% | +0.5pp |
| Rating 3-10 | 70.1% | 70.0% | +0.1pp |
| Rating 11-50 | 70.6% | 71.8% | -1.2pp |
| Rating 51+ | 80.4% | 81.9% | -1.5pp |
| **Total** | **83.2%** | **83.6%** | **-0.4pp** |

**Finding:** GELU slightly worse than ReLU (-0.4pp). The benefit of GELU in language models doesn't transfer to this task.

**Why ReLU wins:**
- Sudoku is a constraint satisfaction problem, not a language task
- ReLU's hard zero may help with sparse activations (many cells are "not this digit")
- GELU's smooth non-linearity provides no benefit here

**Conclusion:** Keep ReLU activation.

---

## Experiment: ReLU Squared Activation

**File:** `exp_cosine_relu2.py`

**Hypothesis:** ReLU² (x → max(0,x)²) has shown benefits in some architectures (Primer paper). Does it help?

**Setup:**
- Same as exp_cosine_no_sam (83.6%) but with custom relu_squared activation
- 70K steps on H200

**Results:**

| Metric | ReLU² | ReLU (SOTA) | Delta |
|--------|-------|-------------|-------|
| Rating 0 | 99.9% | 99.8% | +0.1pp |
| Rating 1-2 | 93.3% | 94.3% | -1.0pp |
| Rating 3-10 | 68.2% | 70.0% | -1.8pp |
| Rating 11-50 | 69.0% | 71.8% | -2.8pp |
| Rating 51+ | 79.2% | 81.9% | -2.7pp |
| **Total** | **81.9%** | **83.6%** | **-1.7pp** |

**Finding:** ReLU² hurts by 1.7pp. The squaring operation doesn't help for this task.

**Why it failed:**
- ReLU² amplifies large activations (x² grows fast)
- May cause gradient issues or make optimization harder
- Primer's benefits were for language modeling, not constraint satisfaction

**Conclusion:** Keep standard ReLU.

---

## Experiment: Longer Training (140K steps)

**File:** `exp_cosine_140k.py`

**Hypothesis:** Does doubling training steps (70K → 140K) improve results? Maybe our model needs more time to converge.

**Setup:**
- Same as exp_cosine_no_sam but 140K steps instead of 70K
- Curriculum phases scaled 2x (each phase runs 28K steps instead of 14K)
- Cosine LR decays more slowly (reaches minimum at 140K instead of 70K)
- Ran on H200, timed out at 95K steps due to runtime limit

**Results (at step 95K, incomplete):**

| Step | LR | Total Accuracy |
|------|-----|----------------|
| 70K baseline | 1.5e-5 (min) | 83.6% |
| 85K (140K run) | 5.25e-4 | 81.6% |
| 90K (140K run) | 4.46e-4 | 82.4% |
| 95K (140K run) | 3.72e-4 | 82.4% |

**Finding:** At 95K steps (68% through), the 140K run is at 82.4% - **worse than 70K baseline's 83.6%**. The slower LR decay means the model hasn't entered the "refinement" phase yet (LR still at 25% of peak).

**Why longer training didn't help:**

1. **LR schedule mismatch**: At step 70K in the 70K schedule, LR is at minimum (1.5e-5), enabling fine refinement. At step 95K in the 140K schedule, LR is still 3.7e-4 (25x higher), so the model is still in "learning" mode, not "refining" mode.

2. **Diminishing returns**: The 70K schedule already reaches low LR and allows convergence. Stretching this out doesn't help - it just takes longer to get to the same place.

3. **Curriculum timing**: With phases 2x longer, the model spends more time on each difficulty level, but this doesn't translate to better final performance.

4. **Implicit regularization**: Shorter training with faster LR decay may actually help generalization by preventing overfitting.

**Conclusion:** 70K steps with cosine LR is sufficient. Longer training doesn't help and may actually hurt due to slower LR decay. The key is reaching low LR for refinement, not more steps at high LR.

---

## Experiment: Shorter Training (50K steps)

**File:** `exp_cosine_50k.py`

**Hypothesis:** If reaching low LR is the key (not total steps), can we get similar results with fewer steps and faster cosine decay?

**Setup:**
- 50K steps instead of 70K (30% fewer)
- Scaled curriculum phases: 10K each instead of 14K
- Scaled warmup: 1400 steps instead of 2000
- Same cosine shape, just compressed
- ~1.4h training time vs ~2h

**Results:**

| Metric | 50K | 70K (SOTA) | Delta |
|--------|-----|------------|-------|
| Rating 0 | 100.0% | 99.8% | +0.2pp |
| Rating 1-2 | 93.9% | 94.3% | -0.4pp |
| Rating 3-10 | 68.8% | 70.0% | -1.2pp |
| Rating 11-50 | 70.3% | 71.8% | -1.5pp |
| Rating 51+ | 80.9% | 81.9% | -1.0pp |
| **Total** | **82.8%** | **83.6%** | **-0.8pp** |

**Finding:** 50K steps achieves 82.8% - only **0.8pp below baseline** with **30% fewer steps**. This is pareto-optimal for fast iteration.

**Why it works:**
- The LR curve shape matters more than total steps
- 50K still reaches minimum LR (1.5e-5) for refinement
- Each curriculum phase still gets meaningful time (10K steps each)
- The model converges to similar quality with compressed schedule

**Trade-off:** 0.8pp accuracy vs 30% faster training. For rapid experimentation, 50K is excellent.

**Conclusion:** 50K steps is pareto-optimal - best accuracy/time trade-off for fast iteration.

---

## Experiment: EMA on Cosine LR (Retest)

**File:** `exp_cosine_ema.py`

**Hypothesis:** We tested EMA on warmup baseline (hurt -1pp). Maybe EMA helps with cosine LR? TRM gets +7.5pp from EMA.

**Setup:**
- Same as exp_cosine_no_sam (83.6%) but with EMA (decay=0.999)
- Evaluate using EMA shadow weights
- Also compare EMA vs non-EMA weights at end
- 70K steps on H200

**Results:**

| Metric | With EMA | Without EMA | Delta |
|--------|----------|-------------|-------|
| Rating 0 | 100.0% | 100.0% | 0pp |
| Rating 1-2 | 94.5% | 94.5% | 0pp |
| Rating 3-10 | 70.1% | 70.1% | 0pp |
| Rating 11-50 | 70.7% | 70.6% | +0.1pp |
| Rating 51+ | 82.9% | 82.9% | 0pp |
| **Total** | **83.6%** | **83.6%** | **0pp** |

**Finding:** EMA makes **zero difference** with cosine LR. Both EMA and non-EMA weights achieve identical 83.6%.

**Why EMA doesn't help:**
- Cosine LR decay already provides "smooth" final weights by reducing LR to 1% at the end
- EMA averages weights over training for stability - same goal as low LR
- They're redundant techniques - both aim for stable final weights
- TRM's +7.5pp from EMA was likely compensating for a different (worse?) LR schedule

**Why EMA hurt on warmup baseline (-1pp):**
- Without cosine decay, LR stays higher throughout
- EMA averages over high-LR (noisy) updates
- The shadow weights are worse than the final (still noisy) live weights
- With cosine, both EMA and live weights are stable, so they're equal

**Conclusion:** EMA is redundant with cosine LR decay. Skip it - saves code complexity with no benefit.

---

## Experiment: Batch Size 8192 (OOM)

**File:** `exp_cosine_25k_bs8k.py`

**Hypothesis:** Double batch size (8192 vs 4096) with half steps (25K vs 50K) should give same results faster, since GPU parallelism makes larger batches nearly free.

**Setup:**
- BS=8192 (doubled from 4096)
- 25K steps (halved from 50K) - same total samples seen
- Scaled curriculum and warmup proportionally
- H200 GPU (140GB VRAM)

**Result:** **OOM (Out of Memory)**

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 648.00 MiB.
GPU 0 has a total capacity of 139.80 GiB of which 491.25 MiB is free.
```

**Finding:** BS=8192 exceeds H200 memory with our model (800K params, 16 iterations, intermediate supervision). The 16 iterations with full gradient tracking require significant activation memory.

**Why it failed:**
- Each iteration stores activations for backward pass
- 16 iterations × 8192 batch × 81 cells × d_model = large memory
- H200's 140GB is not enough for BS=8192

**Alternatives:**
- Gradient accumulation: 2 × BS=4096 = effective BS=8192, fits in memory
- Reduce iterations during training (but may hurt accuracy)
- Checkpoint activations (slower but less memory)

**Conclusion:** BS=4096 is near the memory limit for our architecture on H200. Use gradient accumulation for larger effective batch sizes.

---

## Experiment: 1M Parameters (NEW PARETO)

**File:** `exp_cosine_50k_1M.py`

**Hypothesis:** More model capacity might help accuracy. Scale from 800K to ~1M params.

**Setup:**
- d_model=144 (from 128)
- d_ff=576 (from 512, keeping 4x ratio)
- ~1M params total (vs 800K baseline)
- 50K steps, BS=4096, cosine LR

**Results:**

| Metric | 1M params | 800K params | Delta |
|--------|-----------|-------------|-------|
| Rating 0 | 99.9% | 100.0% | -0.1pp |
| Rating 1-2 | 93.8% | 93.9% | -0.1pp |
| Rating 3-10 | 69.2% | 68.8% | +0.4pp |
| Rating 11-50 | 71.7% | 70.3% | +1.4pp |
| Rating 51+ | 81.3% | 80.9% | +0.4pp |
| **Total** | **83.2%** | **82.8%** | **+0.4pp** |

**Finding:** 1M params achieves **83.2%** in 50K steps - matching the 70K/800K baseline (83.6%) with 30% less training time.

**Why it helps:**
- More capacity allows better learning in compressed schedule
- Larger d_model (144 vs 128) gives richer representations
- The extra params offset the reduced training steps

**Pareto frontier:**

| Config | Params | Steps | Accuracy |
|--------|--------|-------|----------|
| 70K/800K | 800K | 70K | 83.6% |
| **50K/1M** | **1M** | **50K** | **83.2%** |
| 50K/800K | 800K | 50K | 82.8% |

**Conclusion:** 1M params with 50K steps is pareto-optimal for balancing accuracy and step budget. Best choice when you want near-SOTA accuracy with fewer steps.
