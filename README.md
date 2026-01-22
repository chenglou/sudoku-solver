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

## Key Files

- `exp_extreme_baseline.py` - **New baseline**: trains on 2.7M sudoku-extreme (71.4% on extreme, 83.3% on Kaggle)
- `exp_no_x_after_init.py` - Trains on Kaggle data: encodes puzzle once, cleaner architecture (89.9% on Kaggle)
- `exp_recur_add.py` - Trains on Kaggle data: re-encodes puzzle each iteration (90.6% on Kaggle)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `EXPERIMENTS.md` - Full experiment log and results

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
