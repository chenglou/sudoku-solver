# Failure analysis: What puzzles does the model get wrong and why?

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict

torch.set_float32_matmul_precision('high')

# Model architecture (must match training)
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16

ROW_IDX = torch.tensor([i // 9 for i in range(81)])
COL_IDX = torch.tensor([i % 9 for i in range(81)])
BOX_IDX = torch.tensor([(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)])

class SudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(10 + 9, d_model)
        self.row_embed = nn.Embedding(9, d_model)
        self.col_embed = nn.Embedding(9, d_model)
        self.box_embed = nn.Embedding(9, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 9)

    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        device = x.device
        row_idx = ROW_IDX.to(device)
        col_idx = COL_IDX.to(device)
        box_idx = BOX_IDX.to(device)
        pos_embed = self.row_embed(row_idx) + self.col_embed(col_idx) + self.box_embed(box_idx)
        preds = torch.zeros(batch_size, 81, 9, device=device)
        all_logits = []
        all_preds = []
        for _ in range(n_iterations):
            x_in = torch.cat([x, preds], dim=-1)
            h = self.input_proj(x_in)
            h = h + pos_embed
            h = self.transformer(h)
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
            if return_all:
                all_logits.append(logits)
                all_preds.append(preds.clone())
        if return_all:
            return all_logits, all_preds
        return logits


def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x


def check_valid_sudoku(grid):
    """Check if a 9x9 grid is a valid sudoku solution."""
    # Check rows
    for r in range(9):
        if len(set(grid[r])) != 9 or set(grid[r]) != set(range(1, 10)):
            return False
    # Check columns
    for c in range(9):
        col = [grid[r][c] for r in range(9)]
        if len(set(col)) != 9 or set(col) != set(range(1, 10)):
            return False
    # Check boxes
    for box_r in range(3):
        for box_c in range(3):
            box = []
            for r in range(3):
                for c in range(3):
                    box.append(grid[box_r*3 + r][box_c*3 + c])
            if len(set(box)) != 9 or set(box) != set(range(1, 10)):
                return False
    return True


def count_constraint_violations(grid):
    """Count how many constraint violations in a grid."""
    violations = 0
    # Check rows
    for r in range(9):
        row = [grid[r][c] for c in range(9)]
        violations += 9 - len(set(row))
    # Check columns
    for c in range(9):
        col = [grid[r][c] for r in range(9)]
        violations += 9 - len(set(col))
    # Check boxes
    for box_r in range(3):
        for box_c in range(3):
            box = []
            for r in range(3):
                for c in range(3):
                    box.append(grid[box_r*3 + r][box_c*3 + c])
            violations += 9 - len(set(box))
    return violations


device = torch.device("cuda")

# Load model
print("Loading model...")
model = SudokuTransformer().to(device)
state_dict = torch.load("model_curriculum_reverse.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Load test data
print("Loading test data...")
df = pd.read_csv("data/sudoku-3m.csv")

BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 10.0)]
BUCKET_NAMES = ["0.0", "1.x", "2.x", "3.x", "4.x+"]

test_data = {}
for (min_diff, max_diff), name in zip(BUCKETS, BUCKET_NAMES):
    bucket_df = df[(df['difficulty'] >= min_diff) & (df['difficulty'] < max_diff)]
    test_df = bucket_df.tail(500)
    test_data[name] = {
        'puzzles': test_df['puzzle'].tolist(),
        'solutions': test_df['solution'].tolist(),
        'difficulties': test_df['difficulty'].tolist(),
    }
    print(f"  {name}: {len(test_df)} puzzles")

# Analyze each difficulty bucket
print("\n" + "="*70)
print("FAILURE ANALYSIS")
print("="*70)

all_failures = []
all_successes = []

for name in BUCKET_NAMES:
    puzzles = test_data[name]['puzzles']
    solutions = test_data[name]['solutions']
    difficulties = test_data[name]['difficulties']

    # Encode puzzles
    x_test = torch.stack([encode_puzzle(p) for p in puzzles]).to(device)

    with torch.no_grad():
        # Get all iteration predictions
        all_logits, all_preds = model(x_test, return_all=True)
        final_logits = all_logits[-1]
        final_preds = final_logits.argmax(dim=-1).cpu()

    solved = 0
    failures = []
    successes = []

    for b, (puzzle, solution, diff) in enumerate(zip(puzzles, solutions, difficulties)):
        # Build prediction
        pred_solution = list(puzzle)
        wrong_cells = []

        for i in range(81):
            if puzzle[i] == '.':
                pred_digit = str(final_preds[b, i].item() + 1)
                pred_solution[i] = pred_digit
                if pred_digit != solution[i]:
                    wrong_cells.append(i)

        pred_str = ''.join(pred_solution)

        # Collect iteration dynamics for ALL puzzles (success or failure)
        iter_predictions = []
        iter_correct_counts = []

        for iter_idx in range(n_iterations):
            iter_pred = all_preds[iter_idx][b].argmax(dim=-1).cpu()
            iter_solution = []
            correct = 0
            total_holes = 0

            for i in range(81):
                if puzzle[i] == '.':
                    digit = str(iter_pred[i].item() + 1)
                    iter_solution.append(digit)
                    total_holes += 1
                    if digit == solution[i]:
                        correct += 1
                else:
                    iter_solution.append(puzzle[i])

            iter_predictions.append(''.join(iter_solution))
            iter_correct_counts.append(correct)

        if pred_str == solution:
            solved += 1
            successes.append({
                'puzzle': puzzle,
                'solution': solution,
                'difficulty': diff,
                'bucket': name,
                'n_givens': sum(1 for c in puzzle if c != '.'),
                'iter_correct': iter_correct_counts,
                'converged': iter_predictions[-1] == iter_predictions[-2],
            })
        else:
            # Build predicted grid for constraint checking
            pred_grid = [[int(pred_solution[r*9 + c]) for c in range(9)] for r in range(9)]
            violations = count_constraint_violations(pred_grid)

            # Count givens
            n_givens = sum(1 for c in puzzle if c != '.')

            failures.append({
                'puzzle': puzzle,
                'solution': solution,
                'prediction': pred_str,
                'difficulty': diff,
                'bucket': name,
                'n_wrong': len(wrong_cells),
                'wrong_cells': wrong_cells,
                'n_givens': n_givens,
                'violations': violations,
                'iter_correct': iter_correct_counts,
                'converged': iter_predictions[-1] == iter_predictions[-2],  # Last 2 same?
            })

    all_failures.extend(failures)
    all_successes.extend(successes)

    print(f"\n--- {name} ---")
    print(f"Solved: {solved}/500 ({100*solved/500:.1f}%)")
    print(f"Failures: {len(failures)}")

    if failures:
        avg_wrong = np.mean([f['n_wrong'] for f in failures])
        avg_givens = np.mean([f['n_givens'] for f in failures])
        avg_violations = np.mean([f['violations'] for f in failures])
        converged_pct = 100 * sum(1 for f in failures if f['converged']) / len(failures)

        print(f"  Avg cells wrong: {avg_wrong:.1f}")
        print(f"  Avg givens: {avg_givens:.1f}")
        print(f"  Avg constraint violations: {avg_violations:.1f}")
        print(f"  Converged (last 2 iters same): {converged_pct:.1f}%")

# Overall analysis
print("\n" + "="*70)
print("OVERALL PATTERNS")
print("="*70)

print(f"\nTotal: {len(all_successes)} successes, {len(all_failures)} failures")

if all_failures:
    # Cells wrong distribution
    wrong_counts = [f['n_wrong'] for f in all_failures]
    print(f"\nCells wrong distribution:")
    print(f"  1-5 wrong: {sum(1 for w in wrong_counts if 1 <= w <= 5)}")
    print(f"  6-10 wrong: {sum(1 for w in wrong_counts if 6 <= w <= 10)}")
    print(f"  11-20 wrong: {sum(1 for w in wrong_counts if 11 <= w <= 20)}")
    print(f"  21+ wrong: {sum(1 for w in wrong_counts if w > 20)}")

    # Constraint violations
    violation_counts = [f['violations'] for f in all_failures]
    valid_but_wrong = sum(1 for v in violation_counts if v == 0)
    print(f"\nConstraint violations:")
    print(f"  Valid sudoku (0 violations) but wrong: {valid_but_wrong} ({100*valid_but_wrong/len(all_failures):.1f}%)")
    print(f"  Has violations: {len(all_failures) - valid_but_wrong}")

    # Position analysis - which cells are most often wrong?
    cell_wrong_count = defaultdict(int)
    for f in all_failures:
        for cell in f['wrong_cells']:
            cell_wrong_count[cell] += 1

    # Most problematic positions
    sorted_cells = sorted(cell_wrong_count.items(), key=lambda x: -x[1])[:10]
    print(f"\nMost often wrong positions (cell_idx, count):")
    for cell, count in sorted_cells:
        row, col = cell // 9, cell % 9
        box = (row // 3) * 3 + (col // 3)
        print(f"  Cell {cell} (row={row}, col={col}, box={box}): {count} times")

    # Iteration dynamics - compare successes vs failures
    print(f"\nIteration dynamics (avg correct cells per iteration):")
    print(f"  {'Iter':<6} {'Failures':<12} {'Successes':<12}")
    for iter_idx in range(n_iterations):
        avg_correct_fail = np.mean([f['iter_correct'][iter_idx] for f in all_failures])
        avg_correct_succ = np.mean([s['iter_correct'][iter_idx] for s in all_successes]) if all_successes else 0
        print(f"  {iter_idx+1:<6} {avg_correct_fail:<12.1f} {avg_correct_succ:<12.1f}")

    # Convergence analysis
    converged_fail = [f for f in all_failures if f['converged']]
    not_converged_fail = [f for f in all_failures if not f['converged']]
    converged_succ = [s for s in all_successes if s['converged']]
    not_converged_succ = [s for s in all_successes if not s['converged']]

    print(f"\nConvergence (failures):")
    print(f"  Converged to wrong answer: {len(converged_fail)} ({100*len(converged_fail)/len(all_failures):.1f}%)")
    print(f"  Still changing at final iter: {len(not_converged_fail)} ({100*len(not_converged_fail)/len(all_failures):.1f}%)")

    if converged_fail:
        avg_wrong_converged = np.mean([f['n_wrong'] for f in converged_fail])
        print(f"  Avg cells wrong (converged): {avg_wrong_converged:.1f}")
    if not_converged_fail:
        avg_wrong_not_converged = np.mean([f['n_wrong'] for f in not_converged_fail])
        print(f"  Avg cells wrong (not converged): {avg_wrong_not_converged:.1f}")

    if all_successes:
        print(f"\nConvergence (successes):")
        print(f"  Converged to correct answer: {len(converged_succ)} ({100*len(converged_succ)/len(all_successes):.1f}%)")
        print(f"  Still changing at final iter: {len(not_converged_succ)} ({100*len(not_converged_succ)/len(all_successes):.1f}%)")

# Detailed example failures
print("\n" + "="*70)
print("EXAMPLE FAILURES (one per bucket)")
print("="*70)

for name in BUCKET_NAMES:
    bucket_failures = [f for f in all_failures if f['bucket'] == name]
    if bucket_failures:
        # Pick one with median wrong cells
        bucket_failures.sort(key=lambda x: x['n_wrong'])
        example = bucket_failures[len(bucket_failures)//2]

        print(f"\n--- {name} example (median failure) ---")
        print(f"Difficulty: {example['difficulty']:.2f}")
        print(f"Givens: {example['n_givens']}, Wrong cells: {example['n_wrong']}, Violations: {example['violations']}")
        print(f"Converged: {example['converged']}")

        # Show puzzle grid
        print("\nPuzzle:")
        for r in range(9):
            row = example['puzzle'][r*9:(r+1)*9]
            print("  " + " ".join(row))

        print("\nPrediction vs Solution (X = wrong):")
        for r in range(9):
            line = "  "
            for c in range(9):
                idx = r*9 + c
                if example['puzzle'][idx] != '.':
                    line += example['puzzle'][idx] + " "
                elif example['prediction'][idx] == example['solution'][idx]:
                    line += example['prediction'][idx] + " "
                else:
                    line += "X "
            print(line)

        # Show iteration progression for wrong cells
        print(f"\nIteration progression (correct cells): {example['iter_correct']}")
