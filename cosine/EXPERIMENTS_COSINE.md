# Cosine LR Experiments

Experiments investigating cosine LR decay and related training recipe changes. All experiments use reverse curriculum, BS=4096, 70K steps unless noted otherwise. Baseline before cosine: 78.5% (warmup only).

## Cosine LR Decay (SOTA at the time)

**File:** `exp_cosine.py`

After 2K-step linear warmup, LR decays via cosine schedule to 1% of peak (final LR = 1.5e-5).

| Metric | Cosine LR | Warmup Baseline | Delta |
|--------|-----------|-----------------|-------|
| Rating 0 | 99.9% | 99.6% | +0.3pp |
| Rating 1-2 | 94.4% | 90.2% | +4.2pp |
| Rating 3-10 | 70.9% | 62.3% | +8.6pp |
| Rating 11-50 | 74.9% | 66.2% | +8.7pp |
| Rating 51+ | 80.0% | 74.0% | +6.0pp |
| **Total** | **84.0%** | **78.5%** | **+5.5pp** |

**Finding:** Single largest improvement found (+5.5pp). Late-stage LR decay enables fine-tuning without jumping over good solutions.

**Remaining gap with nano-trm:** 84.0% vs 87.4% (3.4pp gap, down from 8.9pp).

---

## Curriculum Variants with Cosine LR

**Files:** `exp_cosine_mixed.py`, `exp_cosine_regular.py`

| Curriculum | Result | vs Reverse |
|------------|--------|------------|
| Reverse (hard→easy) | 84.0% | baseline |
| Mixed (no curriculum) | 83.8% | -0.2pp |
| Regular (easy→hard) | 80.6% | -3.4pp |

**Finding:** Mixed nearly matches reverse with cosine (-0.2pp, vs -2.6pp pre-cosine). Cosine's gradual LR decay provides implicit curriculum structure, reducing the importance of explicit ordering. Regular (easy→hard) still hurts — easy puzzles teach shortcuts.

---

## Cosine LR Without SAM

**File:** `exp_cosine_no_sam.py`

Plain AdamW instead of SAM. **2x faster training** (one forward-backward per step).

| Metric | No SAM | With SAM | Delta |
|--------|--------|----------|-------|
| **Total** | **83.6%** | **84.0%** | **-0.4pp** |

**Finding:** SAM only contributes 0.4pp with cosine (vs +6pp pre-cosine). Both cosine and SAM aim to find flat minima — with cosine doing that job, SAM becomes redundant. **Cosine without SAM is the recommended baseline** (83.6%, ~2h vs ~4h).

---

## Nano-TRM Style Changes (all negative)

Tested several techniques from nano-trm on the cosine no-SAM baseline (83.6%):

| Experiment | File | Change | Result | Delta |
|-----------|------|--------|--------|-------|
| High Weight Decay | `exp_cosine_wd.py` | WD=1.0 (vs 0.01) | 70.1% | **-13.5pp** |
| GELU Activation | `exp_cosine_gelu.py` | GELU instead of ReLU | 83.2% | -0.4pp |
| ReLU Squared | `exp_cosine_relu2.py` | ReLU² activation | 81.9% | -1.7pp |

**Finding:** None of nano-trm's techniques transfer to our smaller model (800K vs 5M params). High WD is catastrophic (-13.5pp) — our model has less capacity to spare. Keep ReLU and WD=0.01.

---

## Training Length

**Files:** `exp_cosine_140k.py`, `exp_cosine_50k.py`

| Config | Steps | Result | vs 70K | Notes |
|--------|-------|--------|--------|-------|
| Longer | 140K | 82.4% (at 95K, incomplete) | -1.2pp | Timed out; slower LR decay means still in "learning" mode at 95K |
| **Shorter** | **50K** | **82.8%** | **-0.8pp** | **30% fewer steps, pareto-optimal for fast iteration** |
| Baseline | 70K | 83.6% | — | |

**Finding:** LR curve shape matters more than total steps. 50K still reaches minimum LR for refinement. Longer training doesn't help — it just takes longer to reach the "refining" phase.

---

## EMA Retest

**File:** `exp_cosine_ema.py`

EMA (decay=0.999) on cosine no-SAM baseline: **0pp difference** (83.6% with or without EMA).

**Finding:** EMA is redundant with cosine LR. Both aim for stable final weights. Pre-cosine EMA hurt (-1pp) because it averaged over noisy high-LR updates. With cosine, both EMA and live weights converge to the same stable point.

---

## Batch Size 8192

**File:** `exp_cosine_25k_bs8k.py`

BS=8192 with 25K steps (same total samples): **OOM on H200** (140GB). 16 iterations × 8192 batch with full gradient tracking exceeds memory. Use gradient accumulation for larger effective batch sizes.

---

## 1M Parameters (Pareto)

**File:** `exp_cosine_50k_1M.py`

d_model=144 (from 128), d_ff=576 (from 512), ~1M params total.

| Config | Params | Steps | Accuracy |
|--------|--------|-------|----------|
| 70K/800K | 800K | 70K | 83.6% |
| **50K/1M** | **1M** | **50K** | **83.2%** |
| 50K/800K | 800K | 50K | 82.8% |

**Finding:** 1M params with 50K steps matches 70K/800K baseline with 30% less training time. More capacity offsets fewer steps. Pareto-optimal for fast iteration.
