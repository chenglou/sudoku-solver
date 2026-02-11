# Neural Sudoku Solver

Experiments on various architectures to solve sudoku

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train SOTA model (auto-downloads sudoku-extreme from HuggingFace)
python iters/exp_baseline_lr2e3.py

# Evaluate at 1024 test-time iterations (98.9%)
python -c "from iters.eval_more_iters import evaluate; evaluate('model_baseline_lr2e3.pt', exp_module='iters.exp_baseline_lr2e3', iter_counts=[1024])"
```

The training code can run on any GPU and provider, agnostically. I'm personally using Modal, with a small wrapper script (`modal_run.py`) that you don't have to use.

## Modal (Optional)

For running on Modal's cloud GPUs:

```sh
pip install modal
modal token new  # authenticate (one-time)

# Train SOTA (detached so it survives terminal close)
modal run --detach modal_run.py --exp iters.exp_baseline_lr2e3

# Monitor progress
modal app logs <app-id>  # app-id shown when you launch

# List/download outputs
modal volume ls sudoku-outputs
modal volume get sudoku-outputs model_baseline_lr2e3.pt .

# Evaluate at 1024 test-time iterations
modal run modal_eval.py --exp iters.exp_baseline_lr2e3 --model model_baseline_lr2e3.pt --iters 1024
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

- `iters/exp_baseline_lr2e3.py` - **SOTA**: 2D RoPE, 800K params, BS=2048, LR=2e-3 (98.9% at 1024 test iters)
- `checkpoint_utils.py` - Checkpoint save/resume utilities (Modal preemption-safe)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `iters/` - Iteration experiments: 32-iter training, adaptive stopping, fixed-point analysis, and [results](iters/EXPERIMENTS_ITERS.md)
- `analyze_failures_new.py` - Per-iteration failure analysis (convergence, oscillation, per-position stats)
- `test_data.py` - Test data loader using test.csv (matches nano-trm for fair comparison)
- `EXPERIMENTS.md` - Full experiment log and results
- `pos_embedding/` - Positional encoding experiments and [results](pos_embedding/EXPERIMENTS_POS.md)
- `muon/` - Muon optimizer experiments and [results](muon/EXPERIMENTS_MUON.md)
- `modal_run.py` - (Optional) Modal wrapper for running experiments on Modal GPUs
- `modal_analyze.py` - (Optional) Modal wrapper for analysis/eval scripts
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

| Model | Params | Training Time | Accuracy |
|-------|--------|---------------|----------|
| **exp_baseline_lr2e3 (1024 test iters)** | 800K | ~2h40m (H200) | **98.9%** |
| exp_baseline_lr2e3 (16 test iters) | 800K | ~2h40m (H200) | 81.8% |
| [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (reference) | 7M | ~18h (L40S) | ~87% |

The model uses sudoku-agnostic 2D RoPE (only knows it's a grid, no constraint structure). Training with BS=2048 produces a model that scales monotonically with test-time iterations — running 1024 iterations at test time (vs 16 during training) yields **98.9%** with no retraining and no collapse. LR is the most important hyperparameter for iteration scaling: LR=2e-3 > 1.5e-3 >> 1e-3 (collapses). See [iters/](iters/EXPERIMENTS_ITERS.md) for the full iteration scaling table, [EXPERIMENTS.md](EXPERIMENTS.md) for detailed analysis, [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) for positional encoding comparisons.
