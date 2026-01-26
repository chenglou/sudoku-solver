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

- `exp_scale_batch_4k.py` - **Current baseline**: 800K params, BS=4096, 70K steps (76.3% on extreme)
- `exp_scale_batch.py` - Previous baseline: 800K params, BS=2048, 70K steps (73.7% on extreme)
- `exp_scale_batch_4k_v2.py` - Reverse curriculum with scaled phases: BS=4096, 10K steps (70.5%)
- `exp_scale_batch_4k_curriculum.py` - Regular curriculum (easy→hard): BS=4096, 10K steps (67.1%)
- `exp_scale_wide.py` - Width scaling experiment: 3.2M params, d=512 (74.8% on extreme)
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

| Model | Params | Accuracy |
|-------|--------|----------|
| **exp_scale_batch_4k** | ~800K | **76.3%** |
| exp_scale_wide | 3.2M | 74.8% |
| exp_extreme_baseline | ~800K | 71.4% |
| exp_no_x_after_init (Kaggle) | ~800K | 31.7% |
| TRM (reference) | 5M | 87.4% |

**Key findings:**
- Batch size scaling is the most efficient lever (BS=4096 beats 8x more params)
- Reverse curriculum (hard→easy) beats regular by +3.4%
- See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed analysis
