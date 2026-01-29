# Shared test data loading - matches nano-trm's approach exactly
# Uses test.csv directly for fair comparison

import csv
import torch
from huggingface_hub import hf_hub_download

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def encode_puzzle(puzzle_str):
    """Encode puzzle string to tensor."""
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.' or c == '0':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x


def load_test_csv(max_per_bucket=5000, device=None):
    """
    Load test data from test.csv (same as nano-trm).

    Returns dict mapping bucket name to:
        {'x': tensor, 'puzzles': list, 'solutions': list}
    """
    test_csv_path = hf_hub_download(
        "sapientinc/sudoku-extreme",
        "test.csv",
        repo_type="dataset"
    )

    # Group by rating bucket
    buckets = {name: {'puzzles': [], 'solutions': [], 'ratings': []}
               for _, _, name in RATING_BUCKETS}

    with open(test_csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for source, question, answer, rating in reader:
            rating = int(rating)
            for min_r, max_r, name in RATING_BUCKETS:
                if min_r <= rating <= max_r:
                    buckets[name]['puzzles'].append(question)
                    buckets[name]['solutions'].append(answer)
                    buckets[name]['ratings'].append(rating)
                    break

    # Sample and encode
    test_data = {}
    for name, data in buckets.items():
        puzzles = data['puzzles']
        solutions = data['solutions']

        if len(puzzles) == 0:
            continue

        # Sample if needed
        if max_per_bucket and len(puzzles) > max_per_bucket:
            import random
            indices = random.sample(range(len(puzzles)), max_per_bucket)
            puzzles = [puzzles[i] for i in indices]
            solutions = [solutions[i] for i in indices]

        # Encode
        x_data = torch.stack([encode_puzzle(p) for p in puzzles])
        if device:
            x_data = x_data.to(device)

        test_data[name] = {
            'x': x_data,
            'puzzles': puzzles,
            'solutions': solutions,
        }

    return test_data


def print_test_stats():
    """Print statistics about test.csv."""
    test_csv_path = hf_hub_download(
        "sapientinc/sudoku-extreme",
        "test.csv",
        repo_type="dataset"
    )

    counts = {name: 0 for _, _, name in RATING_BUCKETS}
    total = 0

    with open(test_csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for source, question, answer, rating in reader:
            rating = int(rating)
            total += 1
            for min_r, max_r, name in RATING_BUCKETS:
                if min_r <= rating <= max_r:
                    counts[name] += 1
                    break

    print(f"test.csv total: {total} puzzles")
    for name, count in counts.items():
        print(f"  Rating {name}: {count}")


if __name__ == "__main__":
    print_test_stats()
