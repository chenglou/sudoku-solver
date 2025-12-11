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
steps = 10000
n_train = 10000
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
df = df[df['difficulty'] == df['difficulty'].min()].head(n_train + n_test)

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

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformer().to(device)
model = torch.compile(model)
x_train = x_train.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            acc = (preds == targets).float().mean().item()
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")

# Test on held-out puzzles (batched to avoid OOM)
x_test = torch.stack([encode_puzzle(p) for p in test_puzzles])
model.eval()
puzzles_correct = 0
cells_correct = 0
cells_total = 0
with torch.no_grad():
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        x_batch = x_test[start:end].to(device)
        logits = model(x_batch)
        preds = logits.argmax(dim=-1).cpu()

        for b, (puzzle, solution) in enumerate(zip(test_puzzles[start:end], test_solutions[start:end])):
            pred_solution = list(puzzle)
            for i in range(81):
                if puzzle[i] == '.':
                    pred_digit = str(preds[b, i].item() + 1)
                    pred_solution[i] = pred_digit
                    cells_total += 1
                    if pred_digit == solution[i]:
                        cells_correct += 1
            pred_solution = ''.join(pred_solution)
            if pred_solution == solution:
                puzzles_correct += 1

print(f"\nTest: {cells_correct}/{cells_total} cells correct ({cells_correct/cells_total:.1%}), {puzzles_correct}/{len(test_puzzles)} puzzles fully solved")
