"""
Modal wrapper for running collapse diagnostics on GPU.

Usage:
    modal run --detach viz/modal_viz.py
"""

import modal

app = modal.App("sudoku-viz")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "logs/", "*.pt", "*.log"])
)


@app.function(
    image=image,
    gpu="H200",
    timeout=2 * 60 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_viz():
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    from viz.plot_collapse_diagnostics import analyze_models

    # Compare stable SOTA vs collapsing models
    configs = [
        ('LR=2e-3 (stable)', '/outputs/model_baseline_lr2e3.pt', 'iters.exp_baseline_lr2e3'),
        ('LR=3e-3 (collapse@64)', '/outputs/model_baseline_lr3e3.pt', 'iters.exp_baseline_lr3e3'),
        ('LR=1e-3 (collapse@128)', '/outputs/model_baseline_lr1e3.pt', 'iters.exp_baseline_lr1e3'),
    ]

    output_dir = "/outputs/viz_diagnostics"
    analyze_models(configs, n_iters=256, n_puzzles=500, device='cuda', output_dir=output_dir)

    outputs_volume.commit()
    print(f"\nPlots saved to volume at viz_diagnostics/")
    print("Download with: modal volume get sudoku-outputs viz_diagnostics/ viz/output/")


@app.local_entrypoint()
def main():
    run_viz.remote()
