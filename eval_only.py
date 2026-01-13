# Eval-only script - loads saved model and evaluates on all difficulties
# Usage: python eval_only.py model_sam_bs512.pt

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

torch.set_float32_matmul_precision('high')

# Hyperparameters (must match training)
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16
batch_size = 512

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

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        row_idx = ROW_IDX.to(device)
        col_idx = COL_IDX.to(device)
        box_idx = BOX_IDX.to(device)
        pos_embed = self.row_embed(row_idx) + self.col_embed(col_idx) + self.box_embed(box_idx)
        preds = torch.zeros(batch_size, 81, 9, device=device)
        for _ in range(n_iterations):
            x_in = torch.cat([x, preds], dim=-1)
            h = self.input_proj(x_in)
            h = h + pos_embed
            h = self.transformer(h)
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
        return logits


def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x


def evaluate_on_difficulty(model, df, difficulty_min, difficulty_max, n_test=1000, skip=0, device='cuda'):
    subset = df[(df['difficulty'] >= difficulty_min) & (df['difficulty'] < difficulty_max)]
    if len(subset) == 0:
        return 0, 0, 0
    # Skip first N puzzles (e.g., training data)
    subset = subset.iloc[skip:]
    if len(subset) == 0:
        return 0, 0, 0
    if len(subset) < n_test:
        n_test = len(subset)
        print(f"  (Only {n_test} puzzles available)")

    subset = subset.head(n_test)
    puzzles = subset['puzzle'].tolist()
    solutions = subset['solution'].tolist()

    x_test = torch.stack([encode_puzzle(p) for p in puzzles]).to(device)

    model.eval()
    puzzles_solved = 0
    total_correct = 0
    total_cells = 0

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_batch = x_test[start:end]
            logits = model(x_batch)
            preds_full = logits.argmax(dim=-1).cpu()

            for b, (puzzle, solution) in enumerate(zip(puzzles[start:end], solutions[start:end])):
                pred_solution = list(puzzle)
                correct = 0
                holes = 0
                for i in range(81):
                    if puzzle[i] == '.':
                        pred_solution[i] = str(preds_full[b, i].item() + 1)
                        holes += 1
                        if pred_solution[i] == solution[i]:
                            correct += 1
                total_correct += correct
                total_cells += holes
                if ''.join(pred_solution) == solution:
                    puzzles_solved += 1

    acc = total_correct / total_cells if total_cells > 0 else 0
    return puzzles_solved, n_test, acc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_only.py <model_checkpoint.pt>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    # Load model
    device = torch.device("cuda")
    model = SudokuTransformer().to(device)
    # Handle torch.compile() prefix in saved state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Strip '_orig_mod.' prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = torch.compile(model)
    print(f"Loaded model from {checkpoint_path}")

    # Load data
    print("Loading data...")
    df = pd.read_csv("data/sudoku-3m.csv")

    # Evaluate on all difficulties
    print("\n" + "="*60)
    print("ZERO-SHOT EVALUATION ON DIFFERENT DIFFICULTIES")
    print("="*60)

    difficulties = [
        (0.0, 0.1, "0.0 (test)", 100000),  # skip first 100k (training data)
        (1.0, 2.0, "1.x", 0),
        (2.0, 3.0, "2.x", 0),
        (3.0, 4.0, "3.x", 0),
        (4.0, 5.0, "4.x", 0),
        (5.0, 10.0, "5.x+", 0),
    ]

    for d_min, d_max, label, skip in difficulties:
        solved, total, acc = evaluate_on_difficulty(model, df, d_min, d_max, n_test=1000, skip=skip, device=device)
        if total > 0:
            print(f"Difficulty {label:12s}: {solved:4d}/{total} solved ({solved/total*100:5.1f}%), cell acc: {acc:.1%}")
        else:
            print(f"Difficulty {label:12s}: No puzzles in this range")
