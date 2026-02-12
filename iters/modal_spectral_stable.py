"""
Modal wrapper for spectral radius analysis + stable model interventions.

Usage:
    modal run --detach iters/modal_spectral_stable.py
"""

import modal

app = modal.App("sudoku-spectral")

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
    timeout=4 * 60 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_analysis():
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    from iters.eval_spectral_radius import analyze_models
    from iters.eval_interventions import evaluate_all

    # Part 1: Spectral radius analysis on all models
    print("=" * 80)
    print("PART 1: SPECTRAL RADIUS ANALYSIS")
    print("=" * 80)

    sr_configs = [
        ('LR=2e-3 d=128 (stable)', '/outputs/model_baseline_lr2e3.pt', 'iters.exp_baseline_lr2e3'),
        ('LR=3e-3 d=128 (collapse@64)', '/outputs/model_baseline_lr3e3.pt', 'iters.exp_baseline_lr3e3'),
        ('LR=1e-3 d=128 (stagnation)', '/outputs/model_baseline_lr1e3.pt', 'iters.exp_baseline_lr1e3'),
        ('d=192 LR=2e-3 (collapse@128)', '/outputs/model_wider_6h_lr2e3.pt', 'iters.exp_wider_6h_lr2e3'),
    ]

    analyze_models(sr_configs, device='cuda', output_dir='/outputs')

    outputs_volume.commit()

    # Part 2: Intervention sweep on stable model
    print("\n" + "=" * 80)
    print("PART 2: INTERVENTIONS ON STABLE MODEL (LR=2e-3, d=128)")
    print("=" * 80)

    evaluate_all('/outputs/model_baseline_lr2e3.pt', 'iters.exp_baseline_lr2e3',
                 device='cuda', output_dir='/outputs')

    outputs_volume.commit()
    print("\nDone. Download logs with:")
    print("  modal volume get sudoku-outputs spectral_radius.log .")
    print("  modal volume get sudoku-outputs model_baseline_lr2e3_interventions.log .")


@app.local_entrypoint()
def main():
    run_analysis.remote()
