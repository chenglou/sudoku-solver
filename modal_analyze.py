"""Modal wrapper for running analyze_failures_new.py on GPU."""

import modal

app = modal.App("sudoku-analyze")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__", "*.pyc", ".git/", "logs/", "*.log"])
)


@app.function(
    image=image,
    gpu="H200",
    timeout=30 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_analysis(exp_name: str, model_filename: str):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root/project")

    from analyze_failures_new import analyze

    model_path = f"/outputs/{model_filename}"
    analyze(model_path, exp_module=exp_name, device='cuda')


@app.function(
    image=image,
    gpu="H200",
    timeout=30 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_more_iters(exp_name: str, model_filename: str, iters: str):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root/project")

    from eval_more_iters import evaluate

    iter_counts = [int(x) for x in iters.split(",")]
    model_path = f"/outputs/{model_filename}"
    evaluate(model_path, exp_module=exp_name, iter_counts=iter_counts, device='cuda')


@app.function(
    image=image,
    gpu="H200",
    timeout=30 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_confidence_stop(exp_name: str, model_filename: str, max_iters: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root/project")

    from eval_confidence_stop import evaluate

    model_path = f"/outputs/{model_filename}"
    evaluate(model_path, exp_module=exp_name, max_iters=max_iters, device='cuda')


@app.function(
    image=image,
    gpu="H200",
    timeout=30 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_fixed_point(exp_name: str, model_filename: str, n_iters: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root/project")

    from eval_fixed_point import evaluate

    model_path = f"/outputs/{model_filename}"
    evaluate(model_path, exp_module=exp_name, n_repeat_iters=n_iters, device='cuda')


@app.local_entrypoint()
def main(
    exp: str = "exp_faster_2drope",
    model: str = "model_faster_2drope.pt",
    mode: str = "analyze",
    iters: str = "16,32,48,64,96,128",
    max_iters: int = 64,
    n_iters: int = 10,
):
    if mode == "fixed_point":
        print(f"Running fixed-point test: exp={exp}, model={model}, n_iters={n_iters}")
        run_fixed_point.remote(exp_name=exp, model_filename=model, n_iters=n_iters)
    elif mode == "analyze":
        print(f"Running analysis: exp={exp}, model={model}")
        run_analysis.remote(exp_name=exp, model_filename=model)
    elif mode == "more_iters":
        print(f"Running more-iters eval: exp={exp}, model={model}, iters={iters}")
        run_more_iters.remote(exp_name=exp, model_filename=model, iters=iters)
    elif mode == "confidence_stop":
        print(f"Running confidence-based stopping eval: exp={exp}, model={model}, max_iters={max_iters}")
        run_confidence_stop.remote(exp_name=exp, model_filename=model, max_iters=max_iters)
