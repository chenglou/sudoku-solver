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
# Used by: eval_extreme.py, exp_extreme_train.py

# Train model (~10 hours on RTX 4090)
python exp_no_x_after_init.py

# Evaluate on sudoku-extreme benchmark
python eval_extreme.py
```

## Key Files

- `exp_no_x_after_init.py` - Recommended: encodes puzzle once, cleaner architecture (89.9%)
- `exp_recur_add.py` - Highest accuracy: re-encodes puzzle each iteration (90.6%)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `EXPERIMENTS.md` - Full experiment log and results

## Results

### Kaggle 3M Dataset (500 test puzzles per difficulty)

| Difficulty | exp_no_x_after_init | exp_recur_add |
|------------|---------------------|---------------|
| 0.x (easy) | 500/500 (100.0%) | 498/500 (99.6%) |
| 1.x | 489/500 (97.8%) | 483/500 (96.6%) |
| 2.x | 452/500 (90.4%) | 461/500 (92.2%) |
| 3.x | 417/500 (83.4%) | 424/500 (84.8%) |
| 4.x+ (hard) | 390/500 (78.0%) | 399/500 (79.8%) |
| **Total** | **2248/2500 (89.9%)** | **2265/2500 (90.6%)** |

### Sudoku-Extreme Benchmark

| Model | Accuracy |
|-------|----------|
| exp_no_x_after_init | 31.7% |
| exp_recur_add | 32.9% |
| TRM (MLP-Mixer) | 87.4% |
