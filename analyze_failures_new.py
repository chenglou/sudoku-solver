# Per-iteration analysis of sudoku solver predictions
# Answers: when does the model converge, what do failures look like,
# and would more iterations help?
#
# Works with any experiment module via --exp flag.
# Currently tested on exp_faster_2drope (2D RoPE baseline).

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


def check_constraints(grid):
    """Check row/col/box constraint violations. grid is length-81 list of ints 1-9."""
    violations = {'row': 0, 'col': 0, 'box': 0}
    for r in range(9):
        row = [grid[r * 9 + c] for c in range(9)]
        violations['row'] += len(row) - len(set(row))
    for c in range(9):
        col = [grid[r * 9 + c] for r in range(9)]
        violations['col'] += len(col) - len(set(col))
    for br in range(3):
        for bc in range(3):
            box = [grid[(br*3+r)*9 + bc*3+c] for r in range(3) for c in range(3)]
            violations['box'] += len(box) - len(set(box))
    return violations


def analyze(model_path, exp_module='exp_faster_2drope', max_test=5000, device='cuda'):
    # Import experiment module for model class and data encoding
    mod = importlib.import_module(exp_module)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = mod.SudokuTransformer().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    n_iters = mod.n_iterations

    # Load test data
    print("Loading test data...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")

    # Group by rating
    buckets = {}
    for i in range(len(test_dataset)):
        r = test_dataset[i]['rating']
        for min_r, max_r, name in RATING_BUCKETS:
            if min_r <= r <= max_r:
                if name not in buckets:
                    buckets[name] = []
                buckets[name].append(i)
                break

    # Cap per bucket
    for name in buckets:
        if len(buckets[name]) > max_test:
            import random
            random.seed(42)
            buckets[name] = random.sample(buckets[name], max_test)

    all_puzzles = []
    all_solutions = []
    all_ratings = []
    all_bucket_names = []
    for name in sorted(buckets.keys(), key=lambda n: [b[2] for b in RATING_BUCKETS].index(n)):
        for idx in buckets[name]:
            all_puzzles.append(test_dataset[idx]['question'])
            all_solutions.append(test_dataset[idx]['answer'])
            all_ratings.append(test_dataset[idx]['rating'])
            all_bucket_names.append(name)

    x_all = mod.encode_puzzles(all_puzzles).to(device)
    n_total = len(all_puzzles)
    print(f"Total test puzzles: {n_total}")

    # Run model and collect per-iteration predictions
    # all_iter_preds[iter][puzzle] = (81,) predicted digits 1-9
    all_iter_preds = [[] for _ in range(n_iters)]
    all_iter_confidence = [[] for _ in range(n_iters)]

    batch_size = 256
    print("Running inference...")
    t_start = time.time()
    use_autocast = device.type == 'cuda'
    ctx = torch.autocast(device.type, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
    with torch.no_grad(), ctx:
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_x = x_all[start:end]
            all_logits = model(batch_x, return_all=True)  # list of n_iters tensors

            for it, logits in enumerate(all_logits):
                probs = F.softmax(logits.float(), dim=-1)
                preds = logits.argmax(dim=-1).cpu()  # (B, 81), values 0-8
                confidence = probs.max(dim=-1).values.cpu()  # (B, 81)
                all_iter_preds[it].append(preds)
                all_iter_confidence[it].append(confidence)
    t_inference = time.time() - t_start
    print(f"Inference time: {t_inference:.1f}s ({n_total} puzzles, {n_total/t_inference:.0f} puzzles/s)")

    # Stack everything
    for it in range(n_iters):
        all_iter_preds[it] = torch.cat(all_iter_preds[it], dim=0)  # (N, 81)
        all_iter_confidence[it] = torch.cat(all_iter_confidence[it], dim=0)  # (N, 81)

    # Build empty cell masks and solution targets
    empty_masks = []
    solution_targets = []  # 0-indexed (0-8)
    for i in range(n_total):
        puzzle = all_puzzles[i]
        solution = all_solutions[i]
        mask = [puzzle[j] == '.' for j in range(81)]
        target = [int(solution[j]) - 1 for j in range(81)]
        empty_masks.append(mask)
        solution_targets.append(target)
    empty_masks = torch.tensor(empty_masks)  # (N, 81) bool
    solution_targets = torch.tensor(solution_targets)  # (N, 81) 0-8

    # ============================================================
    # 1. Per-iteration cell accuracy and puzzle solve rate
    # ============================================================
    print("\n" + "=" * 70)
    print("PER-ITERATION STATS")
    print("=" * 70)
    print(f"{'Iter':>4} | {'Cell Acc':>8} | {'Solved':>8} | {'Avg Conf':>8} | {'Changed':>8}")
    print("-" * 55)

    iter_solved = []
    for it in range(n_iters):
        preds = all_iter_preds[it]  # (N, 81), 0-indexed
        correct = (preds == solution_targets) & empty_masks
        cell_acc = correct.sum().item() / empty_masks.sum().item()

        # Puzzle solved = all empty cells correct
        per_puzzle_correct = correct.sum(dim=1)  # (N,)
        per_puzzle_total = empty_masks.sum(dim=1)  # (N,)
        solved = (per_puzzle_correct == per_puzzle_total)
        n_solved = solved.sum().item()
        iter_solved.append(solved)

        avg_conf = all_iter_confidence[it][empty_masks].mean().item()

        # How many puzzles changed predictions vs previous iteration
        if it > 0:
            changed = (all_iter_preds[it] != all_iter_preds[it - 1]).any(dim=1).sum().item()
        else:
            changed = n_total

        print(f"{it+1:4d} | {cell_acc:7.1%} | {n_solved:5d}/{n_total} | {avg_conf:7.1%} | {changed:5d}/{n_total}")

    # ============================================================
    # 2. First-solve iteration distribution
    # ============================================================
    print("\n" + "=" * 70)
    print("FIRST-SOLVE ITERATION (when puzzles first become fully correct)")
    print("=" * 70)

    first_solve = torch.full((n_total,), -1, dtype=torch.long)
    for it in range(n_iters):
        newly_solved = iter_solved[it] & (first_solve == -1)
        first_solve[newly_solved] = it

    never_solved = (first_solve == -1).sum().item()
    print(f"Never solved: {never_solved}/{n_total}")
    for it in range(n_iters):
        count = (first_solve == it).sum().item()
        if count > 0:
            print(f"  First solved at iter {it+1:2d}: {count:5d}")

    # ============================================================
    # 3. Convergence: when do predictions stop changing?
    # ============================================================
    print("\n" + "=" * 70)
    print("CONVERGENCE (last iteration where predictions changed)")
    print("=" * 70)

    last_change = torch.zeros(n_total, dtype=torch.long)
    for it in range(1, n_iters):
        changed = (all_iter_preds[it] != all_iter_preds[it - 1]).any(dim=1)
        last_change[changed] = it

    final_preds = all_iter_preds[-1]
    final_solved = iter_solved[-1]

    # Split by solved vs failed
    for label, mask in [("Solved", final_solved), ("Failed", ~final_solved)]:
        if mask.sum() == 0:
            continue
        lc = last_change[mask]
        print(f"\n{label} puzzles ({mask.sum().item()}):")
        print(f"  Still changing at iter {n_iters}: {(lc == n_iters - 1).sum().item()}")
        print(f"  Converged before iter {n_iters}: {(lc < n_iters - 1).sum().item()}")
        # Distribution
        for it in range(n_iters):
            count = (lc == it).sum().item()
            if count > 0:
                print(f"    Last changed at iter {it+1:2d}: {count:5d}")

    # ============================================================
    # 4. Failure analysis: how close are failures?
    # ============================================================
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    failed_mask = ~final_solved
    n_failed = failed_mask.sum().item()
    if n_failed > 0:
        failed_preds = final_preds[failed_mask]
        failed_targets = solution_targets[failed_mask]
        failed_empty = empty_masks[failed_mask]

        wrong_cells = ((failed_preds != failed_targets) & failed_empty).sum(dim=1)
        total_empty = failed_empty.sum(dim=1)

        print(f"Failed puzzles: {n_failed}")
        print(f"Wrong cells per failed puzzle:")
        print(f"  Mean: {wrong_cells.float().mean().item():.1f}")
        print(f"  Median: {wrong_cells.float().median().item():.0f}")
        print(f"  Min: {wrong_cells.min().item()}")
        print(f"  Max: {wrong_cells.max().item()}")

        # Distribution of wrong cell counts
        print(f"\nWrong cell count distribution:")
        for threshold in [1, 2, 3, 5, 10, 20, 40]:
            count = (wrong_cells <= threshold).sum().item()
            print(f"  <= {threshold:2d} wrong: {count:5d} ({100*count/n_failed:.1f}%)")

        # Failures by rating bucket
        print(f"\nFailures by rating:")
        for name in sorted(set(all_bucket_names), key=lambda n: [b[2] for b in RATING_BUCKETS].index(n)):
            bucket_mask = torch.tensor([b == name for b in all_bucket_names])
            bucket_total = bucket_mask.sum().item()
            bucket_failed = (bucket_mask & failed_mask).sum().item()
            bucket_solved = bucket_total - bucket_failed
            print(f"  {name:>5}: {bucket_solved}/{bucket_total} solved ({100*bucket_solved/bucket_total:.1f}%)")

        # Constraint violations in failures
        print(f"\nConstraint violations in failures:")
        failed_indices = torch.where(failed_mask)[0]
        total_row_v = 0
        total_col_v = 0
        total_box_v = 0
        zero_violations = 0
        for fi, idx in enumerate(failed_indices):
            puzzle = all_puzzles[idx.item()]
            pred_grid = []
            for j in range(81):
                if puzzle[j] == '.':
                    pred_grid.append(failed_preds[fi][j].item() + 1)
                else:
                    pred_grid.append(int(puzzle[j]))
            v = check_constraints(pred_grid)
            total_row_v += v['row']
            total_col_v += v['col']
            total_box_v += v['box']
            if v['row'] + v['col'] + v['box'] == 0:
                zero_violations += 1

        print(f"  Avg row violations: {total_row_v/n_failed:.1f}")
        print(f"  Avg col violations: {total_col_v/n_failed:.1f}")
        print(f"  Avg box violations: {total_box_v/n_failed:.1f}")
        print(f"  Valid sudoku but wrong answer: {zero_violations}/{n_failed} ({100*zero_violations/n_failed:.1f}%)")

        # Per-position analysis: which cells are most often wrong?
        print(f"\nMost often wrong positions:")
        wrong_by_cell = ((final_preds != solution_targets) & empty_masks)  # (N, 81)
        failed_wrong = wrong_by_cell[failed_mask]  # (n_failed, 81)
        cell_wrong_counts = failed_wrong.sum(dim=0)  # (81,)
        top_cells = cell_wrong_counts.argsort(descending=True)[:10]
        for cell in top_cells:
            c = cell.item()
            count = cell_wrong_counts[c].item()
            if count == 0:
                break
            row, col = c // 9, c % 9
            box = (row // 3) * 3 + (col // 3)
            print(f"  Cell {c:2d} (row={row}, col={col}, box={box}): {count} times wrong")

    # ============================================================
    # 5. "Would more iterations help?" analysis
    # ============================================================
    print("\n" + "=" * 70)
    print("WOULD MORE ITERATIONS HELP?")
    print("=" * 70)

    if n_failed > 0:
        # Check: are failed puzzles still improving (getting more cells right) in late iterations?
        print("Cell accuracy trajectory for FAILED puzzles (last 5 iters):")
        for it in range(max(0, n_iters - 5), n_iters):
            preds = all_iter_preds[it][failed_mask]
            correct = (preds == solution_targets[failed_mask]) & empty_masks[failed_mask]
            cell_acc = correct.sum().item() / empty_masks[failed_mask].sum().item()
            print(f"  Iter {it+1:2d}: {cell_acc:.1%}")

        # Check: did any puzzle go from solved to unsolved between iterations?
        regressions = 0
        for it in range(1, n_iters):
            regressed = iter_solved[it - 1] & ~iter_solved[it]
            regressions += regressed.sum().item()
        print(f"\nSolved→Unsolved regressions across all iterations: {regressions}")

        # Check: predictions still oscillating?
        still_changing = (all_iter_preds[-1] != all_iter_preds[-2]).any(dim=1) & failed_mask
        print(f"Failed puzzles still changing at last iter: {still_changing.sum().item()}/{n_failed}")

    # ============================================================
    # 6. Example failures (one per rating bucket, median wrong count)
    # ============================================================
    print("\n" + "=" * 70)
    print("EXAMPLE FAILURES")
    print("=" * 70)

    if n_failed > 0:
        failed_indices = torch.where(failed_mask)[0]
        failed_wrong_counts = ((final_preds != solution_targets) & empty_masks)[failed_mask].sum(dim=1)

        seen_buckets = set()
        # Sort failures by wrong count to pick median per bucket
        for name in [b[2] for b in RATING_BUCKETS]:
            bucket_failures = []
            for fi, idx in enumerate(failed_indices):
                if all_bucket_names[idx.item()] == name:
                    bucket_failures.append((fi, idx.item(), failed_wrong_counts[fi].item()))
            if not bucket_failures:
                continue

            # Pick median
            bucket_failures.sort(key=lambda x: x[2])
            fi, idx, n_wrong = bucket_failures[len(bucket_failures) // 2]

            puzzle = all_puzzles[idx]
            solution = all_solutions[idx]
            rating = all_ratings[idx]

            print(f"\n--- Rating {name} (median failure, {n_wrong} wrong cells) ---")
            print(f"Rating: {rating}")

            # Show grid: givens as-is, correct predictions as digit, wrong as pred/correct
            print("\nPuzzle → Prediction (X = wrong):")
            for r in range(9):
                line = "  "
                if r % 3 == 0 and r > 0:
                    print("  " + "-" * 17)
                for c in range(9):
                    if c % 3 == 0 and c > 0:
                        line += "| "
                    j = r * 9 + c
                    if puzzle[j] != '.':
                        line += puzzle[j] + " "
                    elif failed_preds[fi][j].item() + 1 == int(solution[j]):
                        line += str(failed_preds[fi][j].item() + 1) + " "
                    else:
                        line += "X "
                print(line)

            # Iteration progression for this puzzle
            iter_correct = []
            for it in range(n_iters):
                correct = ((all_iter_preds[it][idx] == solution_targets[idx]) & empty_masks[idx]).sum().item()
                iter_correct.append(correct)
            total_empty = empty_masks[idx].sum().item()
            print(f"\nIteration progression ({total_empty} empty cells):")
            print(f"  {' '.join(f'{c:3d}' for c in iter_correct)}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .pt file")
    parser.add_argument("--exp", default="exp_faster_2drope", help="Experiment module name")
    parser.add_argument("--max-test", type=int, default=5000, help="Max puzzles per rating bucket")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    analyze(args.model_path, args.exp, args.max_test, args.device)
