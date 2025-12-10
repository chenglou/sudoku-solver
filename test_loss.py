import pandas as pd
import torch
import torch.nn.functional as F

from debug import print_sudoku

df = pd.read_csv("data/sudoku-3m.csv", nrows=10000)
sample = df[df['difficulty'] == df['difficulty'].min()].iloc[1]

puzzle = sample['puzzle']
solution = sample['solution']

# Find holes and their true values
holes = []
targets = []
for i, (p, s) in enumerate(zip(puzzle, solution)):
    if p == '.':
        holes.append(i)
        targets.append(int(s) - 1)  # 0-8

print_sudoku(puzzle)
print_sudoku(solution)

# Predict all 0s for holes
# preds = torch.zeros(len(holes), 9)
pred_indices = torch.tensor([8, 7, 0, 6, 1, 5, 2, 1, 7, 3, 0, 4, 8, 6, 3, 1, 8, 5, 1, 4, 8, 3, 2, 5, 7, 0, 2, 6, 1, 3, 4, 6, 4, 1, 5, 7, 0, 3, 5, 4, 7, 8, 1, 6, 0, 2, 8, 7, 0, 2, 4, 5, 3, 6, 0, 8, 7])
preds = F.one_hot(pred_indices, num_classes=9).float() * 100  # Large logits → near-certain predictions


targets = torch.tensor(targets)
loss = F.cross_entropy(preds, targets)

print(f"Holes: {len(holes)}")
print(f"Preds: {preds}")
print(f"Targets: {targets}")
print(f"Cross entropy loss: {loss.item():.4f}")
