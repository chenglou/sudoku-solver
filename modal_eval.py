"""
Modal wrapper for running eval_more_iters on GPU.

Usage:
    modal run modal_eval.py --exp iters.exp_bs2048_baseline --model model_bs2048_baseline.pt
    modal run modal_eval.py --exp iters.exp_wider_6h --model model_wider_6h.pt --iters '16,32,64,128,256,512,1024'
"""

import modal

app = modal.App("sudoku-eval")

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
    timeout=2 * 60 * 60,  # 2 hours
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_eval(exp_name: str, model_name: str, iter_counts_str: str = "16,32,64,128,256,512,1024"):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    from iters.eval_more_iters import evaluate

    model_path = os.path.join("/outputs", model_name)
    iter_counts = [int(x) for x in iter_counts_str.split(",")]

    evaluate(model_path, exp_module=exp_name, iter_counts=iter_counts, device='cuda',
             output_dir="/outputs")


@app.local_entrypoint()
def main(
    exp: str = "iters.exp_bs2048_baseline",
    model: str = "model_bs2048_baseline.pt",
    iters: str = "16,32,64,128,256,512,1024",
):
    print(f"Evaluating {model} with exp={exp}, iters={iters}")
    run_eval.remote(exp_name=exp, model_name=model, iter_counts_str=iters)
