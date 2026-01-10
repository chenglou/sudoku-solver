# Sudoku Transformer Experiments

All experiments use 100k training steps on easiest difficulty puzzles (100k train, 1k test).

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

**Results:** 96.0% acc, 817 solved (peak at step 96k)

**Finding:** Project-then-add significantly outperforms concat (+3.2% acc, +174 puzzles). Separating projections gives the model more flexibility to learn different representations for given digits vs predictions vs empty indicators.

**Note:** Training was more unstable (fluctuating results) but converged to better final performance.

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

## Summary Table

| Experiment | Acc | Solved | Key Finding |
|------------|-----|--------|-------------|
| Baseline | 92.8% | 643 | - |
| No iteration | 64.3% | 0 | Iteration critical |
| No intermediate | 87.0% | 428 | Intermediate helps |
| No sudoku pos | 88.4% | 409 | Structure helps |
| Project-add | 96.0% | 817 | Better than concat |
| Middle1 (2×2) | 84.2% | 162 | Depth > specialization |
| Middle2 (4×1) | 72.7% | 0 | 1 layer insufficient |
| Unrolled (16×4) | 90.1% | 581 | Weight sharing helps |

---

## Key Insights

1. **Iteration is essential** - Sudoku requires multi-step constraint propagation. Single-pass fails completely.

2. **Depth per iteration matters** - 4 layers needed for effective reasoning. 2 layers marginal, 1 layer broken.

3. **Weight sharing actively helps** - Not just "fine" - it's beneficial! Unrolled model with 16x params performed worse. Weight sharing acts as regularization, forcing a single general iterative function.

4. **Project-then-add beats concat** - Best result so far. Separate projections for different input types.

5. **Intermediate supervision helps** - Gradient signal to all iterations stabilizes training.

6. **Structured pos encoding helps** - But may be redundant with sparse attention (see RRN experiments).
