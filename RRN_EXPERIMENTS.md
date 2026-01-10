# RRN (Recurrent Relational Network) Experiments

All experiments use 100k training steps on easiest difficulty puzzles (100k train, 1k test).

---

## Baseline: RRN with Intermediate Supervision

**File:** `sudoku_rrn.py`

**Architecture:**
- Graph: 81 nodes, ~1620 edges (cells connected if same row/col/box)
- 16 message passing steps with shared weights
- Message MLP: 4-layer (256 → 128)
- Node MLP: 4-layer (384 → 128)
- Simple learned 81-position embedding
- Intermediate supervision (loss at all 16 steps)

**Params:** 194k (vs Transformer's 800k)
**FLOPs:** 2.3B (vs Transformer's 1.1B)
**Speed:** 67ms/batch (vs Transformer's 44ms)

**Results:** 98.2% acc, 869/1000 solved

---

## Ablation: No Structured Positional Encoding

**File:** `rrn_ablation_no_sudoku_pos.py`

**Hypothesis:** Does structured row/col/box embedding help RRN, or does the graph structure already encode this?

**Change:** Use simple learned 81-position embedding instead of row_embed + col_embed + box_embed.

**Results:** 98.2% acc, 869/1000 solved

**Finding:** No difference! The graph edges already encode which cells share constraints. Structured positional encoding is redundant for RRN (unlike transformer where it helped +4.4%).

---

## Ablation: Prediction Feedback (like Transformer)

**File:** `rrn_ablation_pred_feedback.py`

**Hypothesis:** Does feeding softmax predictions back each step help (like transformer does)?

**Change:** Input becomes concat(puzzle, predictions) = 19 dims. Update predictions each step.

**Results:** 97.9% acc, 839/1000 solved

**Finding:** Actually hurts (-0.3% acc, -30 puzzles). Message passing already propagates prediction info implicitly through hidden states. Explicit feedback adds noise.

---

## Ablation: No Intermediate Supervision

**File:** `rrn_ablation_no_intermediate.py`

**Hypothesis:** Does supervising all 16 steps help, or is final-only loss sufficient?

**Change:** Only compute loss on final step output.

**Results:** TODO

---

## Ablation: No Iteration (Single Step)

**File:** `rrn_ablation_no_iteration.py`

**Hypothesis:** Can a single message passing step solve Sudoku?

**Change:** Set num_steps=1.

**Results:** TODO

---

## Ablation: Fewer Steps

**File:** `rrn_ablation_fewer_steps.py`

**Hypothesis:** Is 16 steps necessary? What's the minimum?

**Change:** Try num_steps = 1, 2, 4, 8, 16, 32

**Results:** TODO

---

## Summary Table

| Experiment | Acc | Solved | Key Finding |
|------------|-----|--------|-------------|
| **Baseline** | **98.2%** | **869** | - |
| No structured pos | 98.2% | 869 | Graph makes pos redundant |
| Pred feedback | 97.9% | 839 | Message passing enough |
| No intermediate | TODO | TODO | - |
| No iteration | TODO | TODO | - |
| Fewer steps | TODO | TODO | - |

---

## RRN vs Transformer Comparison

| Model | Params | FLOPs | Speed | Acc | Solved |
|-------|--------|-------|-------|-----|--------|
| **RRN** | **194k** | 2.3B | 67ms | **98.2%** | **869** |
| Transformer | 800k | 1.1B | 44ms | 92.8% | 643 |

RRN wins on accuracy with 4x fewer params, despite 2x more FLOPs and 1.5x slower.

---

## Key Insights (so far)

1. **Graph structure > attention** - Explicit constraint edges outperform learned attention patterns.

2. **Positional encoding redundant** - Graph edges already encode row/col/box relationships.

3. **Prediction feedback unnecessary** - Message passing propagates info implicitly.

4. **Parameter efficient** - 4x fewer params than transformer for better results.

5. **Compute intensive** - Processes 1620 edges per step; slower than dense attention.
