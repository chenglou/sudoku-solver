# Evaluate our model on the sudoku-extreme benchmark
# https://huggingface.co/datasets/sapientinc/sudoku-extreme

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from collections import defaultdict
import numpy as np

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
    """Encode puzzle string to tensor. Handles both '.' and '0' for empty cells."""
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1  # empty cell
        else:
            x[i, int(c)] = 1
    return x


device = torch.device("cuda")

# Load model
print("Loading model...")
model = SudokuTransformer().to(device)
state_dict = torch.load("model_curriculum_reverse.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Load sudoku-extreme test set
print("Loading sudoku-extreme dataset...")
dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
print(f"Test set size: {len(dataset)}")

# Check format
print("\nSample entry:")
sample = dataset[0]
print(f"  source: {sample['source']}")
print(f"  question: {sample['question']}")
print(f"  answer: {sample['answer']}")
print(f"  rating: {sample['rating']}")

# Check rating distribution
ratings = [d['rating'] for d in dataset]
print(f"\nRating distribution:")
print(f"  min: {min(ratings)}, max: {max(ratings)}, mean: {np.mean(ratings):.1f}")

# Define rating buckets
RATING_BUCKETS = [
    (0, 0, "0 (trivial)"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, 1000, "51+"),
]

# Evaluate by rating bucket
print("\n" + "="*60)
print("EVALUATION BY RATING")
print("="*60)

results_by_bucket = defaultdict(lambda: {"total": 0, "solved": 0})

batch_size = 256
questions = dataset['question']
answers = dataset['answer']
ratings = dataset['rating']

# Process in batches
print(f"\nEvaluating {len(dataset)} puzzles...")
for start_idx in range(0, len(dataset), batch_size):
    end_idx = min(start_idx + batch_size, len(dataset))
    batch_questions = questions[start_idx:end_idx]
    batch_answers = answers[start_idx:end_idx]
    batch_ratings = ratings[start_idx:end_idx]

    # Encode batch
    x_batch = torch.stack([encode_puzzle(q) for q in batch_questions]).to(device)

    with torch.no_grad():
        logits = model(x_batch)
        preds = logits.argmax(dim=-1).cpu()

    # Check each puzzle
    for i, (question, answer, rating) in enumerate(zip(batch_questions, batch_answers, batch_ratings)):
        # Build prediction string
        pred_solution = []
        for j in range(81):
            if question[j] == '.' or question[j] == '0':
                pred_solution.append(str(preds[i, j].item() + 1))
            else:
                pred_solution.append(question[j])
        pred_str = ''.join(pred_solution)

        # Find bucket
        for min_r, max_r, bucket_name in RATING_BUCKETS:
            if min_r <= rating <= max_r:
                results_by_bucket[bucket_name]["total"] += 1
                if pred_str == answer:
                    results_by_bucket[bucket_name]["solved"] += 1
                break

    if (start_idx // batch_size) % 50 == 0:
        print(f"  Processed {end_idx}/{len(dataset)}...")

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

total_solved = 0
total_puzzles = 0

for min_r, max_r, bucket_name in RATING_BUCKETS:
    if bucket_name in results_by_bucket:
        data = results_by_bucket[bucket_name]
        total = data["total"]
        solved = data["solved"]
        pct = 100 * solved / total if total > 0 else 0
        print(f"Rating {bucket_name:12s}: {solved:6d}/{total:6d} ({pct:5.1f}%)")
        total_solved += solved
        total_puzzles += total

print("-"*60)
print(f"{'TOTAL':18s}: {total_solved:6d}/{total_puzzles:6d} ({100*total_solved/total_puzzles:5.1f}%)")
