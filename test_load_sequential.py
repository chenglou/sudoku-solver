# Test sequential data loading (current approach)
import torch
import time
from datasets import load_dataset

def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.' or c == '0':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x

def get_targets(puzzle_str, solution_str):
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.' or p == '0':
            holes.append(i)
            targets.append(int(s) - 1)
    return torch.tensor(holes), torch.tensor(targets)

print("Loading dataset...")
dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
train_size = 100000  # Test with 100K for speed

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]

print(f"\nSequential encoding of {train_size} puzzles...")
start = time.time()

train_data = {}
total_puzzles = 0
for min_r, max_r, name in RATING_BUCKETS:
    # Filter indices
    indices = [i for i in range(train_size) if min_r <= dataset[i]['rating'] <= max_r]
    if len(indices) == 0:
        continue

    # Extract and encode
    puzzles = [dataset[i]['question'] for i in indices]
    solutions = [dataset[i]['answer'] for i in indices]
    x_data = torch.stack([encode_puzzle(p) for p in puzzles])
    holes_list = [get_targets(p, s) for p, s in zip(puzzles, solutions)]

    train_data[(min_r, max_r)] = {
        'x': x_data,
        'holes': holes_list,
        'size': len(puzzles),
    }
    total_puzzles += len(puzzles)
    print(f"  {name}: {len(puzzles)} puzzles")

elapsed = time.time() - start
print(f"\nTotal: {total_puzzles} puzzles")
print(f"Time: {elapsed:.2f}s ({total_puzzles/elapsed:.0f} puzzles/sec)")

# Verify data
print(f"\nVerification:")
for (min_r, max_r), data in train_data.items():
    print(f"  Rating {min_r}-{max_r}: x shape {data['x'].shape}, {len(data['holes'])} holes")
