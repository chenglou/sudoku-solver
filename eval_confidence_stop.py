# Test confidence-based early stopping on existing model.
# No retraining needed. Runs all iterations, picks per-puzzle the iteration
# with highest confidence (mean max-softmax over empty cells).
# Also tries: last-before-oscillation, confidence drop detection.

import torch
import torch.nn.functional as F
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



def evaluate(model_path, exp_module='exp_faster_2drope', max_iters=64,
             max_test=5000, device='cuda'):
    mod = importlib.import_module(exp_module)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = mod.SudokuTransformer().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

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
    print(f"Total test puzzles: {n_total}")

    solution_targets = []
    for sol in all_solutions:
        solution_targets.append([int(sol[j]) - 1 for j in range(81)])
    solution_targets = torch.tensor(solution_targets)

    empty_masks = torch.tensor([[p[j] == '.' for j in range(81)] for p in all_puzzles])

    # Run all iterations and collect per-iteration logits
    def run_all_iters(model, x, n_iters):
        """Returns all_logits: list of (B, 81, 9) tensors, one per iteration."""
        batch_size = x.size(0)
        dev = x.device
        rope_cos = mod.ROPE_COS.to(dev)
        rope_sin = mod.ROPE_SIN.to(dev)

        h_prev = model.initial_encoder(x)
        preds = torch.zeros(batch_size, 81, 9, device=dev)
        all_logits = []

        for _ in range(n_iters):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            logits = model.output_head(h)
            preds = F.softmax(logits, dim=-1)
            all_logits.append(logits.cpu())
        return all_logits

    batch_size = 128  # smaller batch since we store all iterations
    print(f"Running {max_iters} iterations, collecting all intermediate outputs...")
    t_start = time.time()

    # Collect all iteration logits: (T, N, 81, 9)
    all_iter_logits = [[] for _ in range(max_iters)]

    use_autocast = device.type == 'cuda'
    ctx = torch.autocast(device.type, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
    with torch.no_grad(), ctx:
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_x = x_all[start:end]
            batch_logits = run_all_iters(model, batch_x, max_iters)
            for t in range(max_iters):
                all_iter_logits[t].append(batch_logits[t])

    # Stack: (T, N, 81, 9)
    all_logits = torch.stack([torch.cat(iter_list, dim=0) for iter_list in all_iter_logits])
    T, N = all_logits.shape[0], all_logits.shape[1]
    t_elapsed = time.time() - t_start
    print(f"Inference done in {t_elapsed:.1f}s\n")

    # Per-iteration predictions: (T, N, 81)
    all_preds = all_logits.argmax(dim=-1)

    # Per-iteration confidence: mean max-softmax over empty cells
    all_probs = F.softmax(all_logits, dim=-1)  # (T, N, 81, 9)
    max_probs = all_probs.max(dim=-1).values    # (T, N, 81)
    # Mask: only empty cells
    empty_masks_expanded = empty_masks.unsqueeze(0).expand(T, -1, -1)  # (T, N, 81)
    masked_probs = max_probs * empty_masks_expanded.float()
    n_empty = empty_masks.sum(dim=1).float()  # (N,)
    confidence = masked_probs.sum(dim=2) / n_empty.unsqueeze(0)  # (T, N)

    def compute_accuracy(preds_per_puzzle, label):
        """preds_per_puzzle: (N, 81)"""
        correct = (preds_per_puzzle == solution_targets) & empty_masks
        per_puzzle_correct = correct.sum(dim=1)
        per_puzzle_total = empty_masks.sum(dim=1)
        solved = (per_puzzle_correct == per_puzzle_total)
        total_solved = solved.sum().item()
        acc = 100 * total_solved / n_total

        bucket_strs = []
        for bname in [b[2] for b in RATING_BUCKETS]:
            mask = torch.tensor([b == bname for b in bucket_names])
            s = (solved & mask).sum().item()
            t = mask.sum().item()
            bucket_strs.append(f"{100*s/t:5.1f}%")

        print(f"  {label:<45} {total_solved:5d}/{n_total} ({acc:5.1f}%) | " +
              " | ".join(bucket_strs))
        return acc

    header = f"  {'Method':<45} {'Solved':>12} | " + " | ".join(f"{b[2]:>5}" for b in RATING_BUCKETS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Baseline: fixed iteration counts
    for fixed_t in [16, 32, 48, 64]:
        if fixed_t <= T:
            compute_accuracy(all_preds[fixed_t - 1], f"Fixed {fixed_t} iters")

    print()

    # Strategy 1: Peak confidence — pick iteration with highest mean confidence per puzzle
    best_iter_conf = confidence.argmax(dim=0)  # (N,)
    peak_conf_preds = all_preds[best_iter_conf, torch.arange(N)]
    compute_accuracy(peak_conf_preds, "Peak confidence (all iters)")

    # Strategy 1b: Peak confidence but only from iterations 16+
    conf_from_16 = confidence.clone()
    conf_from_16[:16] = -1  # mask out first 16
    best_iter_conf_16 = conf_from_16.argmax(dim=0)
    peak_conf_16_preds = all_preds[best_iter_conf_16, torch.arange(N)]
    compute_accuracy(peak_conf_16_preds, "Peak confidence (iters 16+)")

    # Strategy 2: Stop when confidence drops — use iter before first confidence drop
    # For each puzzle, find the first iteration where confidence decreases from previous
    # Use smoothed confidence (window=3) to avoid noise
    window = 3
    if T >= window:
        smoothed = torch.zeros_like(confidence)
        for t in range(T):
            t_start_w = max(0, t - window + 1)
            smoothed[t] = confidence[t_start_w:t+1].mean(dim=0)

        # Find first drop after iter 16
        best_before_drop = torch.full((N,), T - 1, dtype=torch.long)
        for t in range(17, T):
            dropping = (smoothed[t] < smoothed[t-1]) & (best_before_drop == T - 1)
            best_before_drop[dropping] = t - 1

        drop_preds = all_preds[best_before_drop, torch.arange(N)]
        compute_accuracy(drop_preds, "Stop before confidence drop (smoothed, 16+)")

    # Strategy 3: Stop on oscillation — detect when predictions start cycling
    # For each puzzle, check if preds[t] == preds[t-2] (2-cycle)
    best_before_osc = torch.full((N,), T - 1, dtype=torch.long)
    for t in range(18, T):  # start checking after iter 16
        same_as_2ago = (all_preds[t] == all_preds[t-2]).all(dim=1)
        oscillating = same_as_2ago & (best_before_osc == T - 1)
        best_before_osc[oscillating] = t - 2  # use the earlier one

    osc_preds = all_preds[best_before_osc, torch.arange(N)]
    compute_accuracy(osc_preds, "Stop before oscillation (2-cycle, 16+)")

    # Print distribution of selected iterations for peak confidence
    print(f"\n  Peak confidence iter distribution:")
    hist = torch.histc(best_iter_conf.float(), bins=max_iters//4, min=0, max=max_iters-1)
    for i, count in enumerate(hist):
        if count > 0:
            lo = i * 4
            hi = lo + 3
            print(f"    Iters {lo:3d}-{hi:3d}: {int(count):5d} puzzles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .pt file")
    parser.add_argument("--exp", default="exp_faster_2drope")
    parser.add_argument("--max-iters", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.model_path, args.exp, args.max_iters, args.max_test, args.device)
