"""
Modal wrapper for scale-up experiment with true batch size.
Run with: modal run modal_run.py

Outputs (checkpoints, logs) are saved to a Modal volume and synced locally.
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
    .add_local_file("exp_scale_up_big_gpu.py", "/root/exp_scale_up_big_gpu.py")
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,  # more cores for data loading with mp.Pool
    timeout=6 * 60 * 60,  # 6 hours
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_training():
    import os
    import sys

    # Point HuggingFace to the cached volume
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root")
    from exp_scale_up_big_gpu import train

    result = train(output_dir="/outputs")

    # Volumes auto-persist on function exit
    print(f"\nResult: {result}")
    print("\nOutputs saved to 'sudoku-outputs' volume.")
    print("Run 'modal volume ls sudoku-outputs' to see files.")
    print("Run 'modal volume get sudoku-outputs <filename>' to download.")

    return result


@app.local_entrypoint()
def main():
    result = run_training.remote()
    print(f"\nReturned from Modal: {result}")
