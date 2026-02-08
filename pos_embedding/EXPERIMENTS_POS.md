# Positional Encoding Experiments

This document tracks all positional encoding experiments, from the original ablations through the recent sudoku-agnostic alternatives.

## Baseline: Structured row+col+box embeddings

The original architecture uses three separate `nn.Embedding(9, d_model)` tables — one each for row, column, and 3x3 box index. These are summed and added to the residual stream every iteration:

```python
pos_embed = row_embed(row_idx) + col_embed(col_idx) + box_embed(box_idx)
# ...
h = h_prev + pred_proj(preds) + pos_embed  # every iteration
```

This encodes sudoku's constraint structure directly: cells sharing a row, column, or box get related position vectors.

**Baseline accuracy: 82.8%** (sudoku-extreme, 50K steps, BS=4096, cosine LR)

---

## Early ablations (Kaggle / easy-only era)

### Ablation: No Sudoku Positional Encoding

**File:** `ablation_no_sudoku_pos.py`

**Change:** Replace `row_embed + col_embed + box_embed` with simple learned 81-position embedding.

**Results:** 88.4% cell acc, 409 solved (vs baseline 92.8% acc, 643 solved)

**Finding:** Structured positional encoding helps (+4.4% acc, +234 puzzles). It bakes in sudoku structure (which cells share constraints). However, the gap narrows significantly with better training setups (see abspos below).

### Ablation: Sinusoidal Positional Encoding

**File:** `exp_sinusoidal_pos.py`

**Change:** Replace learned `nn.Embedding(9, d_model)` for row/col/box with fixed sinusoidal encodings: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`.

**Results:** 50.1% acc, 0 solved

**Finding:** Complete failure. Standard sinusoidal encoding is designed for long sequences (hundreds+ positions) where the frequencies create distinguishable patterns. For positions 0-8, the sin/cos values are too similar to differentiate. Learned embeddings are much better for small, discrete position spaces.

### RRN Ablation: No Sudoku Positional Encoding

**File:** `rrn_ablation_no_sudoku_pos.py`

**Finding:** In the RRN (Recurrent Relational Network) architecture, structured pos encoding was "not needed" — the message-passing structure already encodes cell relationships.

### Position Embedding Once vs Every Iteration

**File:** `exp_cosine_pos_once.py`

**Hypothesis:** We add pos_embed every iteration. Since h_prev carries forward, is adding it 16x redundant?

**Change:** Add pos_embed only at initialization, not every iteration.

**Results:**

| Metric | Once | Every Iter | Delta |
|--------|------|------------|-------|
| Total | 82.8% | 83.6% | -0.8pp |

**Finding:** Adding pos_embed every iteration helps **+0.8pp**. The repeated position signal acts as a constant anchor — transformer layers can distort position info over iterations. Re-adding pos_embed reinforces "where am I" at each step. Cheap to compute (just addition), so keep it.

---

## Sudoku-agnostic alternatives (sudoku-extreme, 2026-02-07)

All runs: d_model=128, n_heads=4, n_layers=4, n_iterations=16, batch_size=4096, cosine LR, 50K steps, reverse curriculum (hard-to-easy).

The goal: replace the sudoku-specific row/col/box embeddings with more general-purpose positional encodings.

### Row+Col only (no box)

**File:** `exp_faster_rowcol.py`

**Change:** Drop `box_embed`, keep `row_embed + col_embed`. Row and column are just 2D grid coordinates — nothing sudoku-specific.

**Results: 82.6%** (-0.2pp vs baseline)

**Finding:** The box embedding barely matters. 2D grid coordinates do almost all the work.

### 2D RoPE (Rotary Position Embeddings)

**File:** `exp_faster_2drope.py`

**Change:** Replace all additive positional embeddings with 2D Rotary Position Embeddings applied to Q/K in attention. Split head_dim (32) in half: first 16 dims rotated by row index, last 16 by column index. Uses base frequency 10 (not 10000) since positions only range 0-8. Requires custom transformer layer.

**Results: 82.5%** (-0.3pp vs baseline)

**Finding:** Matches baseline within noise. Position info injected through attention rotation instead of additive embeddings. Naturally re-applies every layer and iteration. More generalizable than learned embeddings — works for arbitrary grid sizes without retraining.

### Learned Absolute Position Embedding

**File:** `exp_faster_abspos.py`

**Change:** Replace row/col/box with single `nn.Embedding(81, d_model)` — one learned vector per cell, zero grid knowledge.

**Results: 81.7%** (-1.1pp vs baseline)

**Finding:** Respectable. The old -4.4pp gap (early ablation) shrunk to -1.1pp with better training (cosine LR, curriculum, BS=4096). The model can learn positions from scratch, just slightly less efficiently.

### T5-style Relative Position Bias

**File:** `exp_faster_t5bias.py`

**Change:** No additive position embeddings. Instead, add a learned scalar bias to attention logits based on 1D relative distance (i-j). `nn.Embedding(161, 1)` — 161 = 2*80+1 possible relative distances. Shared across all heads. Passed via `src_mask` to `nn.TransformerEncoder`.

**Results: 73.9%** (-8.9pp vs baseline)

**Finding:** Significantly worse. Two compounding issues: (1) 1D flattened distance is a poor metric for a 2D grid — cells (0,8) and (1,0) are adjacent on the grid but 1 apart in 1D, while same-column cells are 9 apart; (2) passing a float mask to `nn.TransformerEncoder` disables the SDPA fast path, resulting in ~2x slower training. The model effectively got half the training.

### ALiBi (Attention with Linear Biases)

**File:** `exp_faster_alibi.py`

**Change:** No positional embeddings at all. Fixed per-head linear decay bias on 1D flattened distance: `bias[h,i,j] = -slope_h * |i-j|`. Slopes: geometric series [0.25, 0.0625, 0.0156, 0.0039]. Requires custom transformer layer.

**Results: 18.7%** (-64.1pp vs baseline)

**Finding:** Near-total failure. Fixed 1D linear decay cannot encode 2D grid structure. Same fundamental problem as T5 bias (1D distance on a 2D grid) but worse because the bias is fixed rather than learned.

---

## Summary

| Experiment | Accuracy | vs Baseline | Position method | Grid knowledge |
|------------|----------|-------------|-----------------|----------------|
| **row+col+box** (baseline) | **82.8%** | — | Learned additive | Sudoku-specific |
| **Row+col only** | **82.6%** | -0.2pp | Learned additive | Grid only |
| **2D RoPE** | **82.5%** | -0.3pp | Q/K rotation | Grid only |
| Abs pos (81) | 81.7% | -1.1pp | Learned additive | None |
| T5 rel bias | 73.9% | -8.9pp | Learned attn bias (1D) | None |
| Sinusoidal | 50.1%* | — | Fixed additive | Sudoku-specific |
| ALiBi | 18.7% | -64.1pp | Fixed attn bias (1D) | None |

*Sinusoidal tested in early era with different training setup, not directly comparable.

## Key Insights

1. **Box embedding is nearly worthless** — dropping it costs only 0.2pp. The row+col grid structure does all the work.

2. **2D grid knowledge matters a lot** — methods that understand 2D (rowcol, 2D RoPE) score 82.5-82.6%. Methods using 1D distance (T5, ALiBi) score 18-74%. The 2D→1D flattening destroys spatial relationships.

3. **Learned vs fixed barely matters if you have the right structure** — rowcol (learned) and 2D RoPE (fixed rotations) are within 0.1pp. What matters is knowing it's a 2D grid, not whether the encoding is learned.

4. **Position info every iteration helps** — +0.8pp from re-adding pos_embed each iteration. RoPE and ALiBi naturally satisfy this (position is in attention, re-applied every layer/iteration).

5. **Sinusoidal fails for small position spaces** — positions 0-8 are too close in standard sin/cos. Learned embeddings or RoPE with tuned base frequency work much better.

6. **Training speed matters** — T5 bias's float mask disabled flash attention, making it ~2x slower. The 73.9% result is partly a training budget issue.

## Current choice

**2D RoPE** is the new baseline: matches row+col+box within noise (-0.3pp) while being fully sudoku-agnostic. It only assumes a 2D grid, generalizes to arbitrary grid sizes, and naturally re-injects position info every layer/iteration.

## Notes on log locations

- Modal logs are stored in the `sudoku-outputs` volume (download with `modal volume get`).
- Experiment names on volume: `exp_faster_abspos.log`, `exp_faster_t5bias.log`, `exp_faster_rowcol.log`, `exp_faster_2drope.log`, `exp_faster_alibi.log`.
