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
# For Kaggle data:
python exp_no_x_after_init.py

# For sudoku-extreme data (new baseline):
python exp_extreme_baseline.py

# Evaluate on sudoku-extreme benchmark
python eval_extreme.py
```

The training code can run on any GPU and provider, agnostically. I'm personally using Modal, with a small wrapper script (`modal_run.py`) that you don't have to use.

## Modal (Optional)

For running on Modal's cloud GPUs:

```sh
pip install modal
modal token new  # authenticate (one-time)

# Run experiment (detached so it survives terminal close)
modal run --detach modal_run.py --exp exp_scale_batch_4k

# Monitor progress
modal app logs <app-id>  # app-id shown when you launch

# List/download outputs
modal volume ls sudoku-outputs
modal volume get sudoku-outputs model_scale_batch_4k.pt .
```

Experiments must have a `train(output_dir=".")` function. Modal deps are in `requirements-modal.txt` (minimal, no local CUDA).

## TensorBoard

Convert experiment logs to TensorBoard format:

```sh
python logs_to_tensorboard.py
tensorboard --logdir runs/
# Open http://localhost:6006
```

## Key Files

- `exp_faster.py` - **Baseline**: 2D RoPE, 800K params, BS=4096, cosine LR (82.5%)
- `exp_cosine_no_sam.py` - Previous best with sudoku pos encoding: cosine LR, no SAM (83.6%)
- `exp_cosine.py` - Highest accuracy (sudoku pos): cosine LR + SAM (84.0%)
- `checkpoint_utils.py` - Checkpoint save/resume utilities (Modal preemption-safe)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `test_data.py` - Test data loader using test.csv (matches nano-trm for fair comparison)
- `EXPERIMENTS.md` - Full experiment log and results
- `pos_embedding/` - Positional encoding experiments and [results](pos_embedding/EXPERIMENTS_POS.md)
- `muon/` - Muon optimizer experiments and [results](muon/EXPERIMENTS_MUON.md)
- `modal_run.py` - (Optional) Modal wrapper for running experiments on Modal GPUs
- `requirements-modal.txt` - Minimal deps for Modal (no local CUDA libraries)
- `logs_to_tensorboard.py` - Convert experiment logs to TensorBoard format (post-hoc)
- `tensorboard_utils.py` - Real-time TensorBoard logging utility for experiments

## Test Data

For fair comparison with nano-trm, use `test_data.py` which loads test.csv directly:

```python
from test_data import load_test_csv
test_data = load_test_csv(max_per_bucket=5000, device=device)
# Returns dict: {'0': {'x': tensor, 'puzzles': [...], 'solutions': [...]}, '1-2': {...}, ...}
```

## Results (sudoku-extreme benchmark)

| Model | Params | Pos Encoding | GPU | Accuracy |
|-------|--------|--------------|-----|----------|
| **exp_faster** (baseline) | 800K | 2D RoPE | H200 | **82.5%** |
| exp_cosine_no_sam | 800K | row+col+box | H200 | 83.6% |
| exp_cosine | 800K | row+col+box | H200 | 84.0% |
| [nano-trm](https://github.com/olivkoch/nano-trm) (reference) | 5M | — | — | 87.4% |

The baseline uses sudoku-agnostic 2D RoPE (only knows it's a grid, no constraint structure). See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed analysis, [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) for positional encoding comparisons.
