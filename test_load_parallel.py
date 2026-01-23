# Test parallelized data loading
import torch
import time
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

def encode_puzzle_fast(puzzle_str):
    """Vectorized puzzle encoding"""
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.' or c == '0':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x

def encode_batch(puzzles):
    """Encode a batch of puzzles"""
    return torch.stack([encode_puzzle_fast(p) for p in puzzles])

def get_targets_batch(puzzle_solution_pairs):
    """Get targets for a batch"""
    results = []
    for puzzle_str, solution_str in puzzle_solution_pairs:
        holes = []
        targets = []
        for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
            if p == '.' or p == '0':
                holes.append(i)
                targets.append(int(s) - 1)
        results.append((torch.tensor(holes), torch.tensor(targets)))
    return results

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

print(f"\nParallel encoding of {train_size} puzzles...")
start = time.time()

# Step 1: Batch extract ratings (much faster than per-item access)
print("  Extracting ratings in batch...")
t0 = time.time()
# Access all items at once using dataset slicing
batch = dataset[:train_size]
ratings = batch['rating']
questions = batch['question']
answers = batch['answer']
print(f"    Batch extraction: {time.time()-t0:.2f}s")

# Step 2: Categorize by rating using numpy for speed
t0 = time.time()
ratings_np = np.array(ratings)
train_data = {}
total_puzzles = 0

for min_r, max_r, name in RATING_BUCKETS:
    mask = (ratings_np >= min_r) & (ratings_np <= max_r)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        continue

    puzzles = [questions[i] for i in indices]
    solutions = [answers[i] for i in indices]

    # Encode puzzles (still sequential for now - torch.stack needs list)
    x_data = torch.stack([encode_puzzle_fast(p) for p in puzzles])
    holes_list = [get_targets_batch([(p, s)])[0] for p, s in zip(puzzles, solutions)]

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
