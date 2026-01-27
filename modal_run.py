"""
Modal wrapper for running experiments on GPU.

Usage:
    modal run --detach modal_run.py --exp exp_scale_up_big_gpu
    modal run --detach modal_run.py --exp exp_scale_wide

Outputs (checkpoints, logs) are saved to a Modal volume.
"""

import modal

app = modal.App("sudoku-solver")

# Volume for HuggingFace cache (persists across runs, avoids re-downloading)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

# Volume for outputs (checkpoints, logs)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "logs/", "*.pt", "*.log"])
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,  # more cores for data loading with mp.Pool
    timeout=24 * 60 * 60,  # 24 hours (max)
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_training(exp_name: str):
    import os
    import sys
    import importlib

    # Point HuggingFace to the cached volume
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root/project")

    # Dynamically import the experiment module
    exp_module = importlib.import_module(exp_name)

    result = exp_module.train(output_dir="/outputs")

    # Volumes auto-persist on function exit
    print(f"\nResult: {result}")
    print("\nOutputs saved to 'sudoku-outputs' volume.")
    print("Run 'modal volume ls sudoku-outputs' to see files.")
    print("Run 'modal volume get sudoku-outputs <filename>' to download.")

    return result


@app.local_entrypoint()
def main(exp: str = "exp_scale_up_big_gpu"):
    print(f"Running experiment: {exp}")
    result = run_training.remote(exp_name=exp)
    print(f"\nReturned from Modal: {result}")
