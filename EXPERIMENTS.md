# Sudoku Transformer Experiments

Early experiments use 100k training steps on easiest difficulty puzzles (100k train, 1k test).
Later experiments (curriculum, recurrence) use full 2.7M training set across all difficulties (2.5k test).

**Training infrastructure:** Early experiments ran on RTX 4090 (~6h for 70K steps at BS=512). Later experiments (BS=4096, scale_wide, scale_up) ran on Modal H200 (~4h for 70K steps at BS=4096).

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

**File:** `ablation_no_sudoku_pos.py`

**Hypothesis:** Do structured row/col/box embeddings help, or can the model learn positions from scratch?

**Change:** Replace `row_embed + col_embed + box_embed` with simple learned 81-position embedding.

**Results:** 88.4% acc, 409 solved

**Finding:** Structured positional encoding helps (+4.4% acc, +234 puzzles). It bakes in Sudoku structure (which cells share constraints).

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

**File:** `exp_sinusoidal_pos.py`

**Hypothesis:** Would fixed sinusoidal encodings work as well as learned embeddings for row/col/box positions?

**Change:** Replace learned `nn.Embedding(9, d_model)` for row/col/box with fixed sinusoidal encodings using the standard formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`.

**Results:** 50.1% acc, 0 solved

**Finding:** Complete failure. Standard sinusoidal encoding is designed for long sequences (hundreds+ positions) where the frequencies create distinguishable patterns. For positions 0-8, the sin/cos values are too similar to differentiate. Learned embeddings are much better for small, discrete position spaces.

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
| No sudoku pos | 1k easy | 409 | Structure helps |
| Project-add | 1k easy | 555 | No reliable gain |
| Project-concat | 1k easy | 582 | No reliable gain |
| Middle1 (2×2) | 1k easy | 162 | Depth > specialization |
| Middle2 (4×1) | 1k easy | 0 | 1 layer insufficient |
| Unrolled (16×4) | 1k easy | 581 | Weight sharing helps |
| Sinusoidal pos | 1k easy | 0 | Learned >> sinusoidal |
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

---

## Key Insights

1. **Iteration is essential** - Sudoku requires multi-step constraint propagation. Single-pass fails completely.

2. **Depth per iteration matters** - 4 layers needed for effective reasoning. 2 layers marginal, 1 layer broken.

3. **Weight sharing actively helps** - Not just "fine" - it's beneficial! Unrolled model with 16x params performed worse. Weight sharing acts as regularization, forcing a single general iterative function.

4. **Simple concat input is fine** - Fancy projection schemes (project-then-add, project-then-concat) showed high variance and no reliable gains. Simpler is better.

5. **Intermediate supervision helps** - Gradient signal to all iterations stabilizes training.

6. **Structured pos encoding helps** - But may be redundant with sparse attention (see RRN experiments).

7. **Learned embeddings >> sinusoidal for small spaces** - Standard sinusoidal encoding fails catastrophically for positions 0-8. The frequencies are designed for long sequences; for small discrete spaces, learned embeddings adapt much better.

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

20. **Model size is NOT the bottleneck** - Scaling from 800K to 5M params (matching nano-trm) actually made results worse (69.7% vs 71.4%). Combined with the MLP-Mixer result (71.9% ≈ 71.4%), we've ruled out both architecture type and model size as explanations for nano-trm's 87.4% advantage. Data augmentation (digit relabeling) was tested and didn't help. **LR warmup helps +2.2pp** (76.3% → 78.5%). Still investigating: EMA, cosine LR decay, non-learned hidden state init (std=1.0), carry across batches, no-grad warmup cycles.

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

Still need to test: EMA, cosine LR decay, non-learned hidden state init (std=1.0), carry across batches, no-grad warmup cycles.
