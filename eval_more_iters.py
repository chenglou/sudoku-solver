# Test existing model with more iterations at inference time.
# No retraining needed — just override n_iterations during eval.

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import importlib
import time
from datasets import load_dataset

torch.set_float32_matmul_precision('high')

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def evaluate(model_path, exp_module='exp_faster_2drope', iter_counts=[16, 32, 64, 128],
             max_test=5000, device='cuda'):
    mod = importlib.import_module(exp_module)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = mod.SudokuTransformer().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    # Load test data
    print("Loading test data...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")

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

    all_puzzles = []
    all_solutions = []
    bucket_names = []
    for name in sorted(buckets.keys(), key=lambda n: [b[2] for b in RATING_BUCKETS].index(n)):
        for idx in buckets[name]:
            all_puzzles.append(test_dataset[idx]['question'])
            all_solutions.append(test_dataset[idx]['answer'])
            bucket_names.append(name)

    x_all = mod.encode_puzzles(all_puzzles).to(device)
    n_total = len(all_puzzles)
    print(f"Total test puzzles: {n_total}\n")

    # Build solution targets
    solution_targets = []
    for sol in all_solutions:
        solution_targets.append([int(sol[j]) - 1 for j in range(81)])
    solution_targets = torch.tensor(solution_targets)

    empty_masks = torch.tensor([[p[j] == '.' for j in range(81)] for p in all_puzzles])

    # Custom forward that allows overriding iteration count
    def run_with_iters(model, x, n_iters):
        batch_size = x.size(0)
        dev = x.device
        rope_cos = mod.ROPE_COS.to(dev)
        rope_sin = mod.ROPE_SIN.to(dev)

        h_prev = model.initial_encoder(x)
        preds = torch.zeros(batch_size, 81, 9, device=dev)

        for _ in range(n_iters):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            logits = model.output_head(h)
            preds = F.softmax(logits, dim=-1)
        return logits

    batch_size = 256
    print(f"{'Iters':>5} | {'Total Solved':>12} | {'Acc':>6} | {'Time':>6} | " +
          " | ".join(f"{b[2]:>5}" for b in RATING_BUCKETS))
    print("-" * 80)

    for n_iters in iter_counts:
        all_preds = []
        t_start = time.time()

        use_autocast = device.type == 'cuda'
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
        with torch.no_grad(), ctx:
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                batch_x = x_all[start:end]
                logits = run_with_iters(model, batch_x, n_iters)
                all_preds.append(logits.argmax(dim=-1).cpu())

        t_elapsed = time.time() - t_start
        all_preds = torch.cat(all_preds, dim=0)

        # Check puzzle-level accuracy
        correct = (all_preds == solution_targets) & empty_masks
        per_puzzle_correct = correct.sum(dim=1)
        per_puzzle_total = empty_masks.sum(dim=1)
        solved = (per_puzzle_correct == per_puzzle_total)

        # Per-bucket results
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

        print(f"{n_iters:5d} | {total_solved:5d}/{n_total} | {acc:5.1f}% | {t_elapsed:5.1f}s | " +
              " | ".join(bucket_strs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .pt file")
    parser.add_argument("--exp", default="exp_faster_2drope")
    parser.add_argument("--iters", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128])
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.model_path, args.exp, args.iters, args.max_test, args.device)
