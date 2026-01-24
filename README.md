# Neural Sudoku Solver

Experiments on various architectures to solve sudoku

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset (choose one)
# Option 1: Our training dataset (3M puzzles from Kaggle)
mkdir -p data
# Download from: https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings
# Place sudoku-3m.csv in data/

# Option 2: Sudoku-extreme benchmark (auto-downloads from HuggingFace)
# Used by: eval_extreme.py, exp_extreme_baseline.py

# Train model
# For Kaggle data (~10 hours on RTX 4090):
python exp_no_x_after_init.py

# For sudoku-extreme data (new baseline, ~6 hours on RTX 4090):
python exp_extreme_baseline.py

# Evaluate on sudoku-extreme benchmark
python eval_extreme.py
```

The training code can run on any GPU and provider, agnostically. I'm personally using Modal, with a small wrapper script (`modal_run.py`) that you don't have to use.

## Key Files

- `exp_scale_batch.py` - **Current baseline**: 800K params, BS=2048 (73.7% on extreme)
- `exp_extreme_baseline.py` - Previous baseline: 800K params, BS=512 (71.4% on extreme)
- `exp_scale_wide.py` - Width scaling experiment: 3.2M params, d=512 (74.8% on extreme)
- `exp_scale_up_big_gpu.py` - Depth+width scaling: 6.3M params, d=256, L=8 (73.5% on extreme)
- `checkpoint_utils.py` - Checkpoint save/resume utilities (Modal preemption-safe)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `EXPERIMENTS.md` - Full experiment log and results
- `modal_run.py` - (Optional) Modal wrapper for running experiments on Modal GPUs

## Results

### Cross-Dataset Comparison

| Trained on | Data | Kaggle test | sudoku-extreme test |
|------------|------|-------------|---------------------|
| Kaggle | 2.7M | **89.9%** | 31.7% |
| sudoku-extreme | 400K | 81.3% | 63.4% |
| **sudoku-extreme** | **2.7M** | 83.3% | **71.4%** |
| TRM (reference) | 1K | - | 87.4% |

**Key finding:** Domain match matters more than data quantity. Training on sudoku-extreme (400K) beats Kaggle (2.7M) for sudoku-extreme eval despite 7x less data.

### Kaggle 3M Dataset (500 test puzzles per difficulty)

Trained on Kaggle data:

| Difficulty | exp_no_x_after_init | exp_recur_add |
|------------|---------------------|---------------|
| 0.x (easy) | 500/500 (100.0%) | 498/500 (99.6%) |
| 1.x | 489/500 (97.8%) | 483/500 (96.6%) |
| 2.x | 452/500 (90.4%) | 461/500 (92.2%) |
| 3.x | 417/500 (83.4%) | 424/500 (84.8%) |
| 4.x+ (hard) | 390/500 (78.0%) | 399/500 (79.8%) |
| **Total** | **2248/2500 (89.9%)** | **2265/2500 (90.6%)** |

### Sudoku-Extreme Benchmark

| Model | Training Data | Accuracy |
|-------|---------------|----------|
| **exp_extreme_baseline** | sudoku-extreme 2.7M | **71.4%** |
| exp_no_x_after_init | Kaggle 2.7M | 31.7% |
| exp_recur_add | Kaggle 2.7M | 32.9% |
| TRM (MLP-Mixer) | sudoku-hard 1K | 87.4% |

### Scaling Experiments

All experiments trained on sudoku-extreme 2.7M for 70K steps.

| Experiment | Params | Architecture | Accuracy |
|------------|--------|--------------|----------|
| exp_extreme_baseline | ~800K | d=128, L=4, BS=512 | 71.4% |
| **exp_scale_batch** | ~800K | d=128, L=4, BS=2048 | **73.7%** |
| exp_scale_up_big_gpu | 6.3M | d=256, L=8, BS=512 | 73.5% |
| exp_scale_wide | 3.2M | d=512, L=4, BS=512 | 74.8% |
| TRM (reference) | 5M | - | 87.4% |

**Key findings:**
- Batch size scaling (73.7%) nearly matches 8x more params (73.5%) with zero extra cost
- Width scaling (d=512) outperforms depth scaling (L=8) for same param budget
- True batch size matters: Scale UP improved from 69.7% (grad accum) to 73.5% (true BS)
