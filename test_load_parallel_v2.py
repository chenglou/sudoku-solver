# Test parallelized data loading v2 - with multiprocessing for encoding
import torch
import time
from datasets import load_dataset
import numpy as np
import multiprocessing as mp

def encode_puzzle_np(puzzle_str):
    """Encode puzzle to numpy (pickle-friendly for multiprocessing)"""
    x = np.zeros((81, 10), dtype=np.float32)
    for i, c in enumerate(puzzle_str):
        if c == '.' or c == '0':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x

def encode_and_targets(args):
    """Encode puzzle and get targets - for multiprocessing"""
    puzzle_str, solution_str = args
    x = encode_puzzle_np(puzzle_str)
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.' or p == '0':
            holes.append(i)
            targets.append(int(s) - 1)
    return x, np.array(holes, dtype=np.int64), np.array(targets, dtype=np.int64)

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

print(f"\nParallel encoding v2 (multiprocessing) of {train_size} puzzles...")
start = time.time()

# Step 1: Batch extract
t0 = time.time()
batch = dataset[:train_size]
ratings = np.array(batch['rating'])
questions = batch['question']
answers = batch['answer']
print(f"  Batch extraction: {time.time()-t0:.2f}s")

# Step 2: Categorize and encode in parallel
train_data = {}
total_puzzles = 0

n_workers = mp.cpu_count()
print(f"  Using {n_workers} workers")

for min_r, max_r, name in RATING_BUCKETS:
    t0 = time.time()
    mask = (ratings >= min_r) & (ratings <= max_r)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        continue

    puzzles = [questions[i] for i in indices]
    solutions = [answers[i] for i in indices]

    # Parallel encoding
    with mp.Pool(n_workers) as pool:
        results = pool.map(encode_and_targets, zip(puzzles, solutions))

    # Unpack results
    x_list = [torch.from_numpy(r[0]) for r in results]
    holes_list = [(torch.from_numpy(r[1]), torch.from_numpy(r[2])) for r in results]
    x_data = torch.stack(x_list)

    train_data[(min_r, max_r)] = {
        'x': x_data,
        'holes': holes_list,
        'size': len(puzzles),
    }
    total_puzzles += len(puzzles)
    print(f"  {name}: {len(puzzles)} puzzles ({time.time()-t0:.2f}s)")

elapsed = time.time() - start
print(f"\nTotal: {total_puzzles} puzzles")
print(f"Time: {elapsed:.2f}s ({total_puzzles/elapsed:.0f} puzzles/sec)")

# Verify data
print(f"\nVerification:")
for (min_r, max_r), data in train_data.items():
    print(f"  Rating {min_r}-{max_r}: x shape {data['x'].shape}, {len(data['holes'])} holes")
