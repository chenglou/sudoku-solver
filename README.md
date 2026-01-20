# Sudoku Solver Transformer

An iterative transformer that solves sudoku puzzles through recurrent refinement.

## Quick Start

```bash
# Setup
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

On our test set (2500 puzzles): **2265/2500 (90.6%)**

On sudoku-extreme benchmark: **32.9%** (vs TRM's 87.4%)
