# Neural Sudoku Solver

Experiments on various architectures to solve sudoku

## Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train baseline (auto-downloads sudoku-extreme from HuggingFace)
python exp_faster_2drope.py

# Evaluate on sudoku-extreme benchmark
python eval_extreme.py
```

The training code can run on any GPU and provider, agnostically. I'm personally using Modal, with a small wrapper script (`modal_run.py`) that you don't have to use.

## Modal (Optional)

For running on Modal's cloud GPUs:

```sh
pip install modal
modal token new  # authenticate (one-time)

# Run baseline (detached so it survives terminal close)
modal run --detach modal_run.py --exp exp_faster_2drope

# Monitor progress
modal app logs <app-id>  # app-id shown when you launch

# List/download outputs
modal volume ls sudoku-outputs
modal volume get sudoku-outputs model_faster_2drope.pt .
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

- `exp_faster_2drope.py` - **Baseline**: 2D RoPE, 800K params, BS=4096, cosine LR (82.5%, 91.1% with adaptive stopping)
- `exp_cosine_no_sam.py` - Previous best with sudoku pos encoding: cosine LR, no SAM (83.6%)
- `exp_cosine.py` - Highest accuracy (sudoku pos): cosine LR + SAM (84.0%)
- `checkpoint_utils.py` - Checkpoint save/resume utilities (Modal preemption-safe)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `eval_more_iters.py` - Test model at different iteration counts (no retraining)
- `eval_confidence_stop.py` - Confidence-based and oscillation-based adaptive stopping (no retraining)
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

| Model | Params | Pos Encoding | GPU | Accuracy |
|-------|--------|--------------|-----|----------|
| **exp_faster_2drope + oscillation stop** | 800K | 2D RoPE | H200 | **91.1%** |
| exp_faster_2drope + peak confidence (oracle) | 800K | 2D RoPE | H200 | 91.5% |
| **exp_faster_2drope** (baseline) | 800K | 2D RoPE | H200 | 82.5% |
| exp_cosine_no_sam | 800K | row+col+box | H200 | 83.6% |
| exp_cosine | 800K | row+col+box | H200 | 84.0% |
| [nano-trm](https://github.com/olivkoch/nano-trm) (reference) | 5M | — | — | 87.4% |

The baseline uses sudoku-agnostic 2D RoPE (only knows it's a grid, no constraint structure). At test time, running extra iterations with per-puzzle oscillation detection (stop when predictions start cycling) yields **91.1%**, surpassing nano-trm with no retraining. Peak confidence selection (oracle, pick best iteration retroactively) reaches 91.5%. See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed analysis, [pos_embedding/](pos_embedding/EXPERIMENTS_POS.md) for positional encoding comparisons.
