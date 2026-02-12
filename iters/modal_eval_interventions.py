"""
Modal wrapper for running test-time intervention sweeps.

Usage:
    modal run --detach iters/modal_eval_interventions.py
"""

import modal

app = modal.App("sudoku-interventions")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "logs/", "*.pt", "*.log"])
)


@app.function(
    image=image,
    gpu="H200",
    timeout=4 * 60 * 60,  # 4 hours (many sweeps)
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_interventions():
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    from iters.eval_interventions import evaluate_all

    models = [
        ('/outputs/model_baseline_lr3e3.pt', 'iters.exp_baseline_lr3e3'),
        ('/outputs/model_wider_6h_lr2e3.pt', 'iters.exp_wider_6h_lr2e3'),
    ]

    for model_path, exp in models:
        model_name = os.path.basename(model_path).replace('.pt', '')
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*80}")
        evaluate_all(model_path, exp, device='cuda', output_dir='/outputs')

    outputs_volume.commit()
    print("\nDone. Download logs with:")
    print("  modal volume get sudoku-outputs model_baseline_lr3e3_interventions.log .")
    print("  modal volume get sudoku-outputs model_wider_6h_lr2e3_interventions.log .")


@app.local_entrypoint()
def main():
    run_interventions.remote()
