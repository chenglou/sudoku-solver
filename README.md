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

# Train best model (~10 hours on RTX 4090)
python exp_recur_add.py

# Evaluate on sudoku-extreme benchmark
python eval_extreme.py
```

## Key Files

- `exp_recur_add.py` - Best model: iterative transformer with hidden state recurrence
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `EXPERIMENTS.md` - Full experiment log and results

## Results

### Kaggle 3M Dataset (500 test puzzles per difficulty)

| Difficulty | Solved | Accuracy |
|------------|--------|----------|
| 0.x (easy) | 498/500 | 99.6% |
| 1.x | 483/500 | 96.6% |
| 2.x | 461/500 | 92.2% |
| 3.x | 424/500 | 84.8% |
| 4.x+ (hard) | 399/500 | 79.8% |
| **Total** | **2265/2500** | **90.6%** |

### Sudoku-Extreme Benchmark

**32.9%** (vs TRM's 87.4% with MLP-Mixer)
