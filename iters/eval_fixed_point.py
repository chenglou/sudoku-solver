# Test whether the correct solution is a fixed point of f.
# Feed correct solution as softmax predictions, run one iteration,
# check if the model preserves it or drifts.
# No retraining needed. Uses existing model.

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


def evaluate(model_path, exp_module='exp_faster_2drope', n_repeat_iters=10,
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

    # Build correct solution as one-hot (9-dim softmax targets)
    solution_targets = []
    for sol in all_solutions:
        solution_targets.append([int(sol[j]) - 1 for j in range(81)])
    solution_targets = torch.tensor(solution_targets)  # (N, 81) values 0-8

    correct_onehot = F.one_hot(solution_targets, num_classes=9).float()  # (N, 81, 9)

    empty_masks = torch.tensor([[p[j] == '.' for j in range(81)] for p in all_puzzles])  # (N, 81)

    def run_one_iter_from_preds(model, x, preds_input):
        """Run a single iteration of f starting from given predictions.
        Returns logits (B, 81, 9)."""
        dev = x.device
        rope_cos = mod.ROPE_COS.to(dev)
        rope_sin = mod.ROPE_SIN.to(dev)

        h_prev = model.initial_encoder(x)
        h = h_prev + model.pred_proj(preds_input)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        logits = model.output_head(h)
        return logits

    def run_n_iters_from_preds(model, x, preds_input, n_iters):
        """Run n iterations of f starting from given predictions.
        Returns list of logits, one per iteration."""
        dev = x.device
        rope_cos = mod.ROPE_COS.to(dev)
        rope_sin = mod.ROPE_SIN.to(dev)

        h_prev = model.initial_encoder(x)
        preds = preds_input
        all_logits = []
        for _ in range(n_iters):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            logits = model.output_head(h)
            preds = F.softmax(logits, dim=-1)
            all_logits.append(logits)
        return all_logits

    batch_size = 256
    print(f"\n=== Fixed-point test: feed correct solution, run {n_repeat_iters} iterations ===\n")
    t_start = time.time()

    # Collect per-iteration results
    all_iter_preds = [[] for _ in range(n_repeat_iters)]  # (T, N, 81)
    all_iter_confidence = [[] for _ in range(n_repeat_iters)]  # (T, N)

    use_autocast = device.type == 'cuda'
    ctx = torch.autocast(device.type, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
    with torch.no_grad(), ctx:
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_x = x_all[start:end]
            batch_correct = correct_onehot[start:end].to(device)

            batch_logits_list = run_n_iters_from_preds(model, batch_x, batch_correct, n_repeat_iters)
            for t, logits in enumerate(batch_logits_list):
                preds_t = logits.argmax(dim=-1).cpu()  # (B, 81)
                probs_t = F.softmax(logits, dim=-1)
                max_probs_t = probs_t.max(dim=-1).values.cpu()  # (B, 81)
                all_iter_preds[t].append(preds_t)
                all_iter_confidence[t].append(max_probs_t)

    # Stack results
    all_preds = [torch.cat(p, dim=0) for p in all_iter_preds]  # list of (N, 81)
    all_conf = [torch.cat(c, dim=0) for c in all_iter_confidence]  # list of (N, 81)
    t_elapsed = time.time() - t_start
    print(f"Done in {t_elapsed:.1f}s\n")

    # Analyze: after each iteration, how many cells changed from correct?
    print(f"  {'Iter':>5} | {'Cells correct':>13} | {'Puzzles solved':>14} | {'Avg confidence':>15} | {'Cells flipped':>13}")
    print(f"  " + "-" * 75)

    for t in range(n_repeat_iters):
        preds_t = all_preds[t]  # (N, 81)
        conf_t = all_conf[t]    # (N, 81)

        # Per empty cell: is prediction still correct?
        correct_cells = (preds_t == solution_targets) & empty_masks  # (N, 81)
        total_empty = empty_masks.sum().item()
        n_correct = correct_cells.sum().item()
        cell_acc = 100 * n_correct / total_empty

        # Per puzzle: all empty cells correct?
        per_puzzle_correct = correct_cells.sum(dim=1)
        per_puzzle_total = empty_masks.sum(dim=1)
        solved = (per_puzzle_correct == per_puzzle_total)
        n_solved = solved.sum().item()

        # Mean confidence on empty cells
        masked_conf = (conf_t * empty_masks.float()).sum() / empty_masks.sum().float()

        # Cells that flipped from correct
        n_flipped = total_empty - n_correct

        print(f"  {t:5d} | {n_correct:7d}/{total_empty} ({cell_acc:5.1f}%) | {n_solved:6d}/{n_total} ({100*n_solved/n_total:5.1f}%) | {masked_conf:.4f}         | {n_flipped:13d}")

    # Detailed analysis: which cells get flipped? By bucket
    print(f"\n  === Per-bucket analysis (after 1 iteration from correct solution) ===\n")
    preds_1 = all_preds[0]
    flipped = (preds_1 != solution_targets) & empty_masks  # (N, 81)

    header = f"  {'Bucket':>6} | {'Puzzles':>8} | {'Cells flipped':>13} | {'Puzzles preserved':>18} | {'Avg flips/puzzle':>16}"
    print(header)
    print(f"  " + "-" * (len(header) - 2))

    for bname in [b[2] for b in RATING_BUCKETS]:
        mask = torch.tensor([b == bname for b in bucket_names])
        n_puzzles = mask.sum().item()
        if n_puzzles == 0:
            continue
        bucket_flipped = flipped[mask]  # (n_puzzles, 81)
        bucket_empty = empty_masks[mask]
        total_flips = bucket_flipped.sum().item()
        preserved = (bucket_flipped.sum(dim=1) == 0).sum().item()
        avg_flips = total_flips / n_puzzles

        print(f"  {bname:>6} | {n_puzzles:>8} | {total_flips:>13} | {preserved:>10}/{n_puzzles} ({100*preserved/n_puzzles:5.1f}%) | {avg_flips:>16.2f}")

    # How many cells flip TO what?
    print(f"\n  === What do flipped cells become? (after 1 iter from correct) ===")
    flip_mask = flipped  # (N, 81)
    if flip_mask.sum() > 0:
        correct_vals = solution_targets[flip_mask]  # correct digit (0-8)
        pred_vals = preds_1[flip_mask]              # predicted digit (0-8)

        # Distribution of digit changes
        print(f"\n  Total flipped cells: {flip_mask.sum().item()}")

        # Are flipped predictions confident or uncertain?
        flip_conf = all_conf[0][flip_mask]
        print(f"  Mean confidence on flipped cells: {flip_conf.mean():.4f}")
        print(f"  Mean confidence on preserved cells: {all_conf[0][~flipped & empty_masks].mean():.4f}")

        # How far off are the flips? (digit distance)
        digit_diff = (pred_vals - correct_vals).abs().float()
        print(f"  Mean digit distance of flips: {digit_diff.mean():.2f}")
    else:
        print(f"  No cells flipped! Correct solution IS a fixed point.")

    # Multi-iteration stability: does it diverge further or converge back?
    print(f"\n  === Multi-iteration stability (starting from correct) ===")
    print(f"  {'Iter':>5} | {'Cells still correct':>20} | {'Puzzles still solved':>21}")
    print(f"  " + "-" * 55)
    for t in range(n_repeat_iters):
        preds_t = all_preds[t]
        correct_cells = (preds_t == solution_targets) & empty_masks
        total_empty = empty_masks.sum().item()
        n_correct = correct_cells.sum().item()
        per_puzzle_correct = correct_cells.sum(dim=1)
        per_puzzle_total = empty_masks.sum(dim=1)
        solved = (per_puzzle_correct == per_puzzle_total)
        n_solved = solved.sum().item()
        print(f"  {t:5d} | {n_correct:>10}/{total_empty} ({100*n_correct/total_empty:5.1f}%) | {n_solved:>10}/{n_total} ({100*n_solved/n_total:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .pt file")
    parser.add_argument("--exp", default="exp_faster_2drope")
    parser.add_argument("--n-iters", type=int, default=10,
                        help="Number of iterations to run from the correct solution")
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.model_path, args.exp, args.n_iters, args.max_test, args.device)
