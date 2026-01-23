# Verify sequential and parallel produce identical results
import torch
import numpy as np
from datasets import load_dataset
import multiprocessing as mp

def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.' or c == '0':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x

def encode_puzzle_np(puzzle_str):
    x = np.zeros((81, 10), dtype=np.float32)
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

def encode_and_targets(args):
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
test_size = 1000  # Small test for verification

RATING_BUCKETS = [(0, 0, "0"), (1, 2, "1-2"), (3, 10, "3-10"), (11, 50, "11-50"), (51, 1000, "51+")]

# Sequential method
print("\nSequential encoding...")
seq_data = {}
for min_r, max_r, name in RATING_BUCKETS:
    indices = [i for i in range(test_size) if min_r <= dataset[i]['rating'] <= max_r]
    if not indices:
        continue
    puzzles = [dataset[i]['question'] for i in indices]
    solutions = [dataset[i]['answer'] for i in indices]
    x_data = torch.stack([encode_puzzle(p) for p in puzzles])
    holes_list = [get_targets(p, s) for p, s in zip(puzzles, solutions)]
    seq_data[(min_r, max_r)] = {'x': x_data, 'holes': holes_list}

# Parallel method
print("Parallel encoding...")
batch = dataset[:test_size]
ratings = np.array(batch['rating'])
questions = batch['question']
answers = batch['answer']

par_data = {}
with mp.Pool(mp.cpu_count()) as pool:
    for min_r, max_r, name in RATING_BUCKETS:
        mask = (ratings >= min_r) & (ratings <= max_r)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        puzzles = [questions[i] for i in indices]
        solutions = [answers[i] for i in indices]
        results = pool.map(encode_and_targets, zip(puzzles, solutions))
        x_list = [torch.from_numpy(r[0]) for r in results]
        holes_list = [(torch.from_numpy(r[1]), torch.from_numpy(r[2])) for r in results]
        x_data = torch.stack(x_list)
        par_data[(min_r, max_r)] = {'x': x_data, 'holes': holes_list}

# Compare
print("\nVerifying match...")
all_match = True
for key in seq_data:
    if key not in par_data:
        print(f"  {key}: MISSING in parallel")
        all_match = False
        continue

    x_match = torch.allclose(seq_data[key]['x'], par_data[key]['x'])

    holes_match = True
    for (h1, t1), (h2, t2) in zip(seq_data[key]['holes'], par_data[key]['holes']):
        if not torch.equal(h1, h2) or not torch.equal(t1, t2):
            holes_match = False
            break

    if x_match and holes_match:
        print(f"  {key}: MATCH ✓")
    else:
        print(f"  {key}: MISMATCH (x={x_match}, holes={holes_match})")
        all_match = False

print(f"\n{'All data matches!' if all_match else 'MISMATCH DETECTED'}")
