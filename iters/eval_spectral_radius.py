# Jacobian spectral radius estimation via power iteration.
# Estimates the dominant eigenvalue magnitude of df/dh at various operating points.
# SR > 1 implies local instability (perturbations grow → oscillatory divergence).
# SR < 1 implies local contractivity (perturbations shrink → stable convergence).
#
# Method: finite-difference JVP with power iteration.
# f(h) = layers(h + pred_proj(softmax(output_head(h))))
#
# Usage (local):
#   python iters/eval_spectral_radius.py model.pt --exp iters.exp_baseline_lr2e3
#
# Usage (Modal):
#   modal run --detach iters/modal_spectral_stable.py

import torch
import torch.nn.functional as F
import numpy as np
import importlib
import argparse
import os
import time
from datasets import load_dataset

torch.set_float32_matmul_precision('high')


def run_and_save_checkpoints(model, exp_mod, x_batch, checkpoints, device):
    """Run model and save h_prev at specified iteration checkpoints."""
    batch_size = x_batch.size(0)
    rope_cos = exp_mod.ROPE_COS.to(device)
    rope_sin = exp_mod.ROPE_SIN.to(device)

    h_prev = model.initial_encoder(x_batch)
    preds = torch.zeros(batch_size, 81, 9, device=device)

    max_iter = max(checkpoints)
    saved = {}

    for i in range(max_iter):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        logits = model.output_head(h)
        preds = F.softmax(logits, dim=-1)

        if (i + 1) in checkpoints:
            saved[i + 1] = h_prev.clone()

    return saved


def iteration_step(model, h, rope_cos, rope_sin):
    """One iteration: f(h) = layers(h + pred_proj(softmax(output_head(h))))."""
    preds = F.softmax(model.output_head(h), dim=-1)
    h_new = h + model.pred_proj(preds)
    for layer in model.layers:
        h_new = layer(h_new, rope_cos, rope_sin)
    return h_new


def estimate_spectral_radius(model, exp_mod, h_star, device,
                              n_power_iters=100, eps=1e-3):
    """Power iteration via finite-difference JVP.

    Returns per-puzzle spectral radius estimates and convergence history.
    """
    batch_size = h_star.size(0)
    rope_cos = exp_mod.ROPE_COS.to(device)
    rope_sin = exp_mod.ROPE_SIN.to(device)

    # Random initial vector, unit norm per puzzle
    v = torch.randn_like(h_star)
    norms = v.reshape(batch_size, -1).norm(dim=-1)
    v = v / norms[:, None, None]

    # Base evaluation at h*
    f_base = iteration_step(model, h_star, rope_cos, rope_sin)

    history = []

    for i in range(n_power_iters):
        # Finite-difference JVP: Jv ≈ (f(h* + εv) - f(h*)) / ε
        f_perturbed = iteration_step(model, h_star + eps * v, rope_cos, rope_sin)
        Jv = (f_perturbed - f_base) / eps

        # Per-puzzle growth rate
        Jv_norms = Jv.reshape(batch_size, -1).norm(dim=-1)
        v_norms = v.reshape(batch_size, -1).norm(dim=-1)
        sigma = Jv_norms / v_norms
        history.append(sigma.cpu())

        # Normalize for next iteration
        v = Jv / Jv_norms[:, None, None]

    return sigma.cpu(), history


def analyze_models(model_configs, checkpoints=None,
                   n_puzzles=50, n_power_iters=100, device='cuda',
                   output_dir=None):
    """
    model_configs: list of (name, model_path, exp_module_name)
    """
    if checkpoints is None:
        checkpoints = [16, 32, 64, 128, 256]

    log_file = None
    if output_dir:
        log_path = os.path.join(output_dir, "spectral_radius.log")
        log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    # Load test data using first model's exp module
    first_exp = importlib.import_module(model_configs[0][2])
    dataset = load_dataset("sapientinc/sudoku-extreme", split="test")

    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), n_puzzles)
    puzzles = [dataset[i]['question'] for i in indices]
    x = first_exp.encode_puzzles(puzzles).to(device)

    log(f"Spectral Radius Analysis")
    log(f"Puzzles: {n_puzzles}, Power iters: {n_power_iters}, eps: 1e-3")
    log(f"Checkpoints (warmup iters): {checkpoints}")
    log("")

    for name, model_path, exp_name in model_configs:
        exp_mod = importlib.import_module(exp_name)
        model = exp_mod.SudokuTransformer().to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        model.eval()

        # Re-encode if different exp module (different architecture)
        x_this = exp_mod.encode_puzzles(puzzles).to(device)

        log(f"{'='*70}")
        log(f"MODEL: {name}")
        log(f"{'='*70}")

        # Run forward and save h at checkpoints
        t_start = time.time()
        with torch.no_grad():
            h_checkpoints = run_and_save_checkpoints(
                model, exp_mod, x_this, checkpoints, device)
        t_warmup = time.time() - t_start
        log(f"  Warmup ({max(checkpoints)} iters): {t_warmup:.1f}s")

        log(f"  {'Iter':>6} | {'Mean SR':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | SR>1")
        log(f"  {'-'*62}")

        for cp in checkpoints:
            h_star = h_checkpoints[cp]
            t_start = time.time()

            with torch.no_grad():
                sr, history = estimate_spectral_radius(
                    model, exp_mod, h_star, device, n_power_iters)

            t_elapsed = time.time() - t_start
            mean_sr = sr.mean().item()
            std_sr = sr.std().item()
            min_sr = sr.min().item()
            max_sr = sr.max().item()
            n_unstable = (sr > 1.0).sum().item()

            log(f"  {cp:6d} | {mean_sr:8.4f} | {std_sr:8.4f} | {min_sr:8.4f} | {max_sr:8.4f} | {n_unstable:3d}/{n_puzzles} ({t_elapsed:.1f}s)")

        log("")
        del model
        torch.cuda.empty_cache()

    if log_file:
        log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to model .pt file')
    parser.add_argument('--exp', default='iters.exp_baseline_lr2e3')
    parser.add_argument('--name', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-puzzles', type=int, default=50)
    parser.add_argument('--n-power-iters', type=int, default=100)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    name = args.name or os.path.basename(args.model_path).replace('.pt', '')
    configs = [(name, args.model_path, args.exp)]
    analyze_models(configs, n_puzzles=args.n_puzzles,
                   n_power_iters=args.n_power_iters,
                   device=args.device, output_dir=args.output_dir)
