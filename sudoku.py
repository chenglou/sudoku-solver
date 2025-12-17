# this is a one-shot transformer model for solving sudoku puzzles. Just a forward pass taking in the original puzzle and outputting the solution
# it predicts each cell at ~75% accuracy (random would be 11%). But with those odds, it still fails to solve any puzzle since a puzzle has many empty cells
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from debug import print_sudoku

# Hyperparameters
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 6
lr = 1e-3
steps = 100000
n_train = 100000
n_test = 1000
batch_size = 512

class SudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(10, d_model)
        self.pos_embed = nn.Parameter(torch.randn(81, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 9)

    def forward(self, x):
        # x: (batch, 81, 10)
        x = self.input_proj(x)  # (batch, 81, d_model)
        x = x + self.pos_embed  # add positional encoding
        x = self.transformer(x)  # (batch, 81, d_model)
        x = self.output_head(x)  # (batch, 81, 9)
        return x

def encode_puzzle(puzzle_str):
    """Convert puzzle string to (81, 10) tensor."""
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1  # empty indicator
        else:
            digit = int(c) - 1  # 0-8
            x[i, 1 + digit] = 1  # one-hot digit
    return x

def get_targets(puzzle_str, solution_str):
    """Return (hole_indices, target_digits) for empty cells."""
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.':
            holes.append(i)
            targets.append(int(s) - 1)  # 0-8
    return torch.tensor(holes), torch.tensor(targets)

df = pd.read_csv("data/sudoku-3m.csv")
easy_df = df[df['difficulty'] == df['difficulty'].min()]
print(f"Puzzles at easiest difficulty: {len(easy_df)} (need {n_train + n_test})")
df = easy_df.head(n_train + n_test)

train_puzzles = df['puzzle'].tolist()[:n_train]
train_solutions = df['solution'].tolist()[:n_train]
test_puzzles = df['puzzle'].tolist()[n_train:]
test_solutions = df['solution'].tolist()[n_train:]

print(f"Train: {len(train_puzzles)}, Test: {len(test_puzzles)} (difficulty={df['difficulty'].iloc[0]})")

# Prepare training data
x_train = torch.stack([encode_puzzle(p) for p in train_puzzles])  # (n_train, 81, 10)
# Per-puzzle hole indices and targets
train_holes = []  # list of (cell_indices, targets) per puzzle
for p, s in zip(train_puzzles, train_solutions):
    cell_idx = []
    targets = []
    for i, (pc, sc) in enumerate(zip(p, s)):
        if pc == '.':
            cell_idx.append(i)
            targets.append(int(sc) - 1)
    train_holes.append((torch.tensor(cell_idx), torch.tensor(targets)))

# Prepare test data
x_test = torch.stack([encode_puzzle(p) for p in test_puzzles])  # (n_test, 81, 10)
test_holes = []
for p, s in zip(test_puzzles, test_solutions):
    cell_idx = []
    targets = []
    for i, (pc, sc) in enumerate(zip(p, s)):
        if pc == '.':
            cell_idx.append(i)
            targets.append(int(sc) - 1)
    test_holes.append((torch.tensor(cell_idx), torch.tensor(targets)))

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformer().to(device)
model = torch.compile(model)
x_train = x_train.to(device)
x_test = x_test.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def evaluate_test():
    """Evaluate on test set, return (loss, accuracy, puzzles_solved)."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_cells = 0
    puzzles_solved = 0

    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_batch = x_test[start:end]
            logits = model(x_batch)

            # Gather holes for this batch
            hole_b, hole_c, targets = [], [], []
            for i, idx in enumerate(range(start, end)):
                cell_idx, tgt = test_holes[idx]
                hole_b.extend([i] * len(cell_idx))
                hole_c.extend(cell_idx.tolist())
                targets.extend(tgt.tolist())
            hole_b = torch.tensor(hole_b, device=device)
            hole_c = torch.tensor(hole_c, device=device)
            targets = torch.tensor(targets, device=device)

            logits_holes = logits[hole_b, hole_c]
            total_loss += F.cross_entropy(logits_holes, targets, reduction='sum').item()
            preds = logits_holes.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_cells += len(targets)

            # Check full puzzle solves
            preds_full = logits.argmax(dim=-1).cpu()
            for b, (puzzle, solution) in enumerate(zip(test_puzzles[start:end], test_solutions[start:end])):
                pred_solution = list(puzzle)
                for i in range(81):
                    if puzzle[i] == '.':
                        pred_solution[i] = str(preds_full[b, i].item() + 1)
                if ''.join(pred_solution) == solution:
                    puzzles_solved += 1

    return total_loss / total_cells, total_correct / total_cells, puzzles_solved

total_holes = sum(len(h[0]) for h in train_holes)
print(f"Total empty cells in train: {total_holes}")
print(f"\nTraining on {device}...")
for step in range(steps):
    model.train()
    optimizer.zero_grad()

    # Sample mini-batch
    batch_idx = torch.randperm(n_train)[:batch_size]
    x_batch = x_train[batch_idx]  # (batch_size, 81, 10)

    logits = model(x_batch)  # (batch_size, 81, 9)

    # Gather holes for this batch
    hole_b, hole_c, targets = [], [], []
    for i, idx in enumerate(batch_idx.tolist()):
        cell_idx, tgt = train_holes[idx]
        hole_b.extend([i] * len(cell_idx))
        hole_c.extend(cell_idx.tolist())
        targets.extend(tgt.tolist())
    hole_b = torch.tensor(hole_b, device=device)
    hole_c = torch.tensor(hole_c, device=device)
    targets = torch.tensor(targets, device=device)

    logits_holes = logits[hole_b, hole_c]
    loss = F.cross_entropy(logits_holes, targets)

    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == steps - 1:
        with torch.no_grad():
            preds = logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()
        train_loss = loss.item()

        # Evaluate on test set every 1000 steps
        if step % 1000 == 0 or step == steps - 1:
            test_loss, test_acc, puzzles_solved = evaluate_test()
            print(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Test Loss: {test_loss:.4f} Acc: {test_acc:.2%} | Solved: {puzzles_solved}/{n_test}")
        else:
            print(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%}")

# Final test evaluation
test_loss, test_acc, puzzles_solved = evaluate_test()
print(f"\nFinal Test: Loss {test_loss:.4f} | Acc {test_acc:.1%} | {puzzles_solved}/{n_test} puzzles solved")
