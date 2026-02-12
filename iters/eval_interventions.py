# Test-time interventions to fix iteration collapse.
# No retraining — just modifies the forward pass during inference.
#
# Interventions:
#   damping:    h_prev = α*h + (1-α)*h_prev  (under-relaxation)
#   pred_scale: h = h_prev + β*pred_proj(preds)  (feedback scaling)
#   pre_norm:   h = norm(h) before output head  (bounds logit growth)
#
# Usage (local):
#   python iters/eval_interventions.py model.pt --exp iters.exp_baseline_lr3e3
#
# Usage (Modal):
#   modal run --detach iters/modal_eval_interventions.py

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import importlib
import time
import os
from datasets import load_dataset

torch.set_float32_matmul_precision('high')

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def run_with_intervention(model, x, n_iters, exp_mod, intervention='none', param=1.0):
    """Forward pass with optional test-time intervention."""
    batch_size = x.size(0)
    dev = x.device
    rope_cos = exp_mod.ROPE_COS.to(dev)
    rope_sin = exp_mod.ROPE_SIN.to(dev)

    h_prev = model.initial_encoder(x)
    preds = torch.zeros(batch_size, 81, 9, device=dev)

    # For pre_norm: capture the norm scale from iteration 16 (training regime)
    norm_scale = None

    for i in range(n_iters):
        # Prediction feedback
        if intervention == 'pred_scale':
            h = h_prev + param * model.pred_proj(preds)
        else:
            h = h_prev + model.pred_proj(preds)

        # Transformer layers
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)

        # Damping: h_prev = α*h + (1-α)*h_prev
        if intervention == 'damping':
            h_prev = param * h + (1.0 - param) * h_prev
        else:
            h_prev = h

        # Pre-output normalization
        if intervention == 'pre_norm':
            # Normalize h to have consistent norm, preserving direction
            h_for_output = F.layer_norm(h, [h.size(-1)])
            logits = model.output_head(h_for_output)
        else:
            logits = model.output_head(h)

        preds = F.softmax(logits, dim=-1)

    return logits


def evaluate_intervention(model, exp_mod, x_all, solution_targets, empty_masks, bucket_names,
                           iter_counts, intervention, param, log_fn, batch_size=256):
    """Run eval with a specific intervention and parameter."""
    n_total = x_all.size(0)

    for n_iters in iter_counts:
        all_preds = []
        t_start = time.time()

        use_autocast = x_all.device.type == 'cuda'
        ctx = torch.autocast(x_all.device.type, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
        with torch.no_grad(), ctx:
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                batch_x = x_all[start:end]
                logits = run_with_intervention(model, batch_x, n_iters, exp_mod,
                                                intervention, param)
                all_preds.append(logits.argmax(dim=-1).cpu())

        t_elapsed = time.time() - t_start
        all_preds = torch.cat(all_preds, dim=0)

        correct = (all_preds == solution_targets) & empty_masks
        per_puzzle_correct = correct.sum(dim=1)
        per_puzzle_total = empty_masks.sum(dim=1)
        solved = (per_puzzle_correct == per_puzzle_total)

        bucket_results = {}
        for name in [b[2] for b in RATING_BUCKETS]:
            mask = torch.tensor([b == name for b in bucket_names])
            bucket_solved = (solved & mask).sum().item()
            bucket_total = mask.sum().item()
            bucket_results[name] = (bucket_solved, bucket_total)

        total_solved = solved.sum().item()
        acc = 100 * total_solved / n_total

        bucket_strs = []
        for name in [b[2] for b in RATING_BUCKETS]:
            s, t = bucket_results[name]
            bucket_strs.append(f"{100*s/t:5.1f}%")

        log_fn(f"{n_iters:5d} | {total_solved:5d}/{n_total} | {acc:5.1f}% | {t_elapsed:5.1f}s | " +
               " | ".join(bucket_strs))


def evaluate_all(model_path, exp_module, device='cuda', output_dir=None):
    """Run all interventions on a single model."""
    log_file = None
    if output_dir:
        model_name = os.path.basename(model_path).replace(".pt", "")
        log_path = os.path.join(output_dir, f"{model_name}_interventions.log")
        log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    mod = importlib.import_module(exp_module)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = mod.SudokuTransformer().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    log("Loading test data...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")

    max_test = 5000
    buckets = {}
    for i in range(len(test_dataset)):
        r = test_dataset[i]['rating']
        for min_r, max_r, name in RATING_BUCKETS:
            if min_r <= r <= max_r:
                if name not in buckets:
                    buckets[name] = []
                buckets[name].append(i)
                break

    import random
    random.seed(42)
    for name in buckets:
        if len(buckets[name]) > max_test:
            buckets[name] = random.sample(buckets[name], max_test)

    all_puzzles, all_solutions, bucket_names = [], [], []
    for name in sorted(buckets.keys(), key=lambda n: [b[2] for b in RATING_BUCKETS].index(n)):
        for idx in buckets[name]:
            all_puzzles.append(test_dataset[idx]['question'])
            all_solutions.append(test_dataset[idx]['answer'])
            bucket_names.append(name)

    x_all = mod.encode_puzzles(all_puzzles).to(device)
    n_total = len(all_puzzles)
    log(f"Total test puzzles: {n_total}")

    solution_targets = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in all_solutions])
    empty_masks = torch.tensor([[p[j] == '.' for j in range(81)] for p in all_puzzles])

    header = (f"{'Iters':>5} | {'Total Solved':>12} | {'Acc':>6} | {'Time':>6} | " +
              " | ".join(f"{b[2]:>5}" for b in RATING_BUCKETS))

    iter_counts = [16, 32, 64, 128, 256, 512, 1024]

    # Baseline (no intervention)
    log(f"\n{'='*80}")
    log(f"BASELINE (no intervention)")
    log(f"{'='*80}")
    log(header)
    log("-" * 80)
    evaluate_intervention(model, mod, x_all, solution_targets, empty_masks, bucket_names,
                          iter_counts, 'none', 1.0, log)

    # Damping sweep
    for alpha in [0.9, 0.8, 0.7, 0.5]:
        log(f"\n{'='*80}")
        log(f"DAMPING α={alpha}")
        log(f"{'='*80}")
        log(header)
        log("-" * 80)
        evaluate_intervention(model, mod, x_all, solution_targets, empty_masks, bucket_names,
                              iter_counts, 'damping', alpha, log)

    # Prediction scaling sweep
    for beta in [0.5, 0.3, 0.1]:
        log(f"\n{'='*80}")
        log(f"PRED_SCALE β={beta}")
        log(f"{'='*80}")
        log(header)
        log("-" * 80)
        evaluate_intervention(model, mod, x_all, solution_targets, empty_masks, bucket_names,
                              iter_counts, 'pred_scale', beta, log)

    # Pre-output normalization
    log(f"\n{'='*80}")
    log(f"PRE_NORM (LayerNorm before output head)")
    log(f"{'='*80}")
    log(header)
    log("-" * 80)
    evaluate_intervention(model, mod, x_all, solution_targets, empty_masks, bucket_names,
                          iter_counts, 'pre_norm', 0, log)

    if log_file:
        log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to model .pt file')
    parser.add_argument('--exp', default='iters.exp_baseline_lr3e3')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    evaluate_all(args.model_path, args.exp, args.device)
