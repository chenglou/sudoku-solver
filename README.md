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

- `exp_cosine_no_sam.py` - **Recommended**: 800K params, BS=4096, cosine LR, no SAM (83.6%, 2x faster)
- `exp_cosine.py` - Highest accuracy: 800K params, BS=4096, cosine LR + SAM (84.0%)
- `exp_scale_batch_4k.py` - Pre-cosine baseline: 800K params, BS=4096, 70K steps (76.3%)
- `exp_scale_batch.py` - Previous baseline: 800K params, BS=2048, 70K steps (73.7% on extreme)
- `exp_scale_batch_4k_v2.py` - Reverse curriculum with scaled phases: BS=4096, 10K steps (70.5%)
- `exp_scale_batch_4k_curriculum.py` - Regular curriculum (easy→hard): BS=4096, 10K steps (67.1%)
- `exp_scale_wide.py` - Width scaling experiment: 3.2M params, d=512 (74.8% on extreme)
- `checkpoint_utils.py` - Checkpoint save/resume utilities (Modal preemption-safe)
- `eval_extreme.py` - Evaluate on sudoku-extreme benchmark
- `test_data.py` - Test data loader using test.csv (matches nano-trm for fair comparison)
- `EXPERIMENTS.md` - Full experiment log and results
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

| Model | Params | Batch Size | GPU | Time | Accuracy |
|-------|--------|------------|-----|------|----------|
| **exp_cosine_no_sam** | 800K | 4096 | H200 | ~2h | **83.6%** |
| exp_cosine | 800K | 4096 | H200 | ~4h | 84.0% |
| exp_scale_wide | 3.2M | 512 | H200 | ~6h | 74.8% |
| exp_extreme_baseline | 800K | 512 | RTX 4090 | ~6h | 71.4% |
| [nano-trm](https://github.com/olivkoch/nano-trm) (reference) | 5M | 256 | - | - | 87.4% |

**Note:** exp_cosine_no_sam is recommended as the default. SAM adds only +0.4pp but doubles training time.

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed analysis and ablations.
