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
steps = 1000

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

# Load one puzzle
df = pd.read_csv("data/sudoku-3m.csv", nrows=10000)
sample = df[df['difficulty'] == df['difficulty'].min()].iloc[1]
puzzle = sample['puzzle']
solution = sample['solution']

print("Training on single puzzle:")
print_sudoku(puzzle)
print("Solution:")
print_sudoku(solution)

# Prepare data
x = encode_puzzle(puzzle).unsqueeze(0)  # (1, 81, 10)
holes, targets = get_targets(puzzle, solution)
print(f"\nEmpty cells: {len(holes)}")

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformer().to(device)
x = x.to(device)
targets = targets.to(device)
holes = holes.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print(f"\nTraining on {device}...")
for step in range(steps):
    model.train()
    optimizer.zero_grad()

    logits = model(x)  # (1, 81, 9)
    logits_holes = logits[0, holes]  # (num_holes, 9)
    loss = F.cross_entropy(logits_holes, targets)

    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == steps - 1:
        with torch.no_grad():
            preds = logits_holes.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")

# Final prediction
model.eval()
with torch.no_grad():
    logits = model(x)
    preds = logits[0].argmax(dim=-1)  # (81,)

    # Build predicted solution
    pred_solution = list(puzzle)
    for i in range(81):
        if puzzle[i] == '.':
            pred_solution[i] = str(preds[i].item() + 1)
    pred_solution = ''.join(pred_solution)

print("\nPredicted solution:")
print_sudoku(pred_solution)
print("\nGround truth:")
print_sudoku(solution)
print(f"\nCorrect: {pred_solution == solution}")
