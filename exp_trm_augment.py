# Experiment: TRM-style digit relabeling augmentation
# TRM trains on only 1K examples but with 1000× digit relabeling augmentation
# This means each puzzle can be transformed 1000 ways by permuting digits 1-9
#
# Key augmentations (from TRM's build_sudoku_dataset.py):
# 1. Digit relabeling: random permutation of 1-9
# 2. Row permutation: shuffle 3 bands, then rows within each band
# 3. Column permutation: shuffle 3 stacks, then columns within each stack
# 4. Transpose: 50% probability
#
# Combined with our nested loop model to see if augmentation helps

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import numpy as np
import os
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, restore_optimizer, save_checkpoint

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "trm_augment_checkpoint_step"

# TRM-style hyperparameters
hidden_size = 512
L_layers = 2
H_cycles = 3
L_cycles = 6
expansion = 4
lr = 1e-4
weight_decay = 1.0
total_steps = 70000
batch_size = 256  # TRM uses 256 (smaller since augmentation provides diversity)

# Data augmentation settings
train_size = 10000    # Use fewer base puzzles (TRM uses 1K)
num_augmentations = 100  # Augmentations per puzzle per epoch (TRM uses 1000)

CONFIG = {
    'experiment': 'exp_trm_augment',
    'hidden_size': hidden_size,
    'L_layers': L_layers,
    'H_cycles': H_cycles,
    'L_cycles': L_cycles,
    'batch_size': batch_size,
    'lr': lr,
    'train_size': train_size,
    'num_augmentations': num_augmentations,
}

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """
    TRM-style sudoku augmentation.

    Applies three transformations that preserve sudoku validity:
    1. Digit relabeling: permute digits 1-9
    2. Position permutation: shuffle bands/rows and stacks/columns
    3. Transpose: 50% probability

    Args:
        board: 9x9 numpy array (0 for empty, 1-9 for digits)
        solution: 9x9 numpy array (1-9 for all cells)

    Returns:
        augmented_board, augmented_solution
    """
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.concatenate([[0], np.random.permutation(np.arange(1, 10))])

    # Randomly decide whether to transpose
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks)
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T.copy()
        # Apply the position mapping
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion=4):
        super().__init__()
        inner_size = int(hidden_size * expansion * 2 / 3)
        inner_size = (inner_size + 63) // 64 * 64
        self.w1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.w2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.w3 = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TRMBlock(nn.Module):
    def __init__(self, hidden_size, seq_len=81, expansion=4):
        super().__init__()
        self.mlp_t = SwiGLU(seq_len, expansion)
        self.mlp = SwiGLU(hidden_size, expansion)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_t = self.mlp_t(x_t)
        x = rms_norm(x + x_t.transpose(1, 2))
        x = rms_norm(x + self.mlp(x))
        return x


class ReasoningModule(nn.Module):
    def __init__(self, hidden_size, L_layers, seq_len=81, expansion=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, seq_len, expansion)
            for _ in range(L_layers)
        ])

    def forward(self, hidden_states, input_injection):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TRMSudoku(nn.Module):
    def __init__(self, hidden_size, L_layers, H_cycles, L_cycles, expansion=4, seq_len=81):
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        self.embed_tokens = nn.Embedding(10, hidden_size)
        self.embed_scale = hidden_size ** 0.5

        self.L_level = ReasoningModule(hidden_size, L_layers, seq_len, expansion)

        self.H_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(hidden_size) * 0.02)

        self.lm_head = nn.Linear(hidden_size, 9, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        input_embeddings = self.embed_tokens(x) * self.embed_scale

        z_H = self.H_init.expand(batch_size, 81, -1)
        z_L = self.L_init.expand(batch_size, 81, -1)

        with torch.no_grad():
            for _H_step in range(self.H_cycles - 1):
                for _L_step in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings)
                z_H = self.L_level(z_H, z_L)

        for _L_step in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.L_level(z_H, z_L)

        return self.lm_head(z_H)


def parse_puzzle(puzzle_str):
    """Convert puzzle string to 9x9 numpy array."""
    arr = np.zeros((9, 9), dtype=np.int64)
    for i, c in enumerate(puzzle_str):
        if c != '.' and c != '0':
            arr[i // 9, i % 9] = int(c)
    return arr


def puzzle_to_tensor(puzzle_arr):
    """Convert 9x9 numpy array to flat tensor."""
    return torch.tensor(puzzle_arr.flatten(), dtype=torch.long)


def get_targets_from_arr(board_arr, solution_arr):
    """Get hole indices and targets from numpy arrays."""
    board_flat = board_arr.flatten()
    solution_flat = solution_arr.flatten()
    holes = []
    targets = []
    for i in range(81):
        if board_flat[i] == 0:
            holes.append(i)
            targets.append(solution_flat[i] - 1)  # 0-indexed
    return torch.tensor(holes), torch.tensor(targets)


def train(output_dir="."):
    device = torch.device("cuda")

    print("Loading sudoku-extreme train split...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    print(f"Total available: {len(dataset)}")

    # Sample diverse difficulty puzzles
    # TRM uses puzzles from sudoku-hard which are all hard
    # Let's sample from rating 3+ to match TRM's difficulty level
    print(f"\nSampling {train_size} hard puzzles (rating >= 3)...")
    hard_indices = [i for i in range(len(dataset)) if dataset[i]['rating'] >= 3]
    print(f"  Found {len(hard_indices)} hard puzzles")

    if len(hard_indices) < train_size:
        print(f"  Not enough hard puzzles, using all")
        selected_indices = hard_indices
    else:
        selected_indices = random.sample(hard_indices, train_size)

    # Store base puzzles as numpy arrays
    train_puzzles = []
    for idx in selected_indices:
        puzzle_str = dataset[idx]['question']
        solution_str = dataset[idx]['answer']
        board = parse_puzzle(puzzle_str)
        solution = parse_puzzle(solution_str)
        train_puzzles.append((board, solution))
    print(f"  Loaded {len(train_puzzles)} base puzzles")

    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    print(f"Test set: {len(test_dataset)}")

    print("\nPreparing test data...")
    test_data = {}
    for min_r, max_r, name in RATING_BUCKETS:
        indices = [i for i in range(len(test_dataset)) if min_r <= test_dataset[i]['rating'] <= max_r]
        if len(indices) == 0:
            continue
        if len(indices) > 5000:
            indices = random.sample(indices, 5000)
        puzzles = [test_dataset[i]['question'] for i in indices]
        solutions = [test_dataset[i]['answer'] for i in indices]
        x_test = torch.stack([
            torch.tensor([0 if c in '.0' else int(c) for c in p], dtype=torch.long)
            for p in puzzles
        ]).to(device)
        test_data[name] = {
            'x': x_test,
            'puzzles': puzzles,
            'solutions': solutions,
        }
        print(f"  Test {name}: {len(puzzles)} puzzles")

    model = TRMSudoku(
        hidden_size=hidden_size,
        L_layers=L_layers,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        expansion=expansion,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    checkpoint_path, start_step = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    checkpoint_data = None
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    if checkpoint_data:
        restore_optimizer(optimizer, checkpoint_data, device)
        print(f"Resumed from step {start_step}")

    print(f"\nExperiment: TRM with Digit Relabeling Augmentation")
    print(f"Architecture: hidden_size={hidden_size}, L_layers={L_layers}")
    print(f"Iterations: H_cycles={H_cycles} × L_cycles={L_cycles}")
    print(f"Data: {train_size} base puzzles × {num_augmentations} augmentations")
    print(f"Batch size: {batch_size}, lr: {lr}")
    print(f"Output directory: {output_dir}")

    log_path = os.path.join(output_dir, "exp_trm_augment.log")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def sample_augmented_batch(bs):
        """Sample batch with on-the-fly augmentation."""
        x_list = []
        holes_list = []
        counts_list = []

        for _ in range(bs):
            # Pick a random base puzzle
            base_board, base_solution = random.choice(train_puzzles)

            # Apply random augmentation
            aug_board, aug_solution = shuffle_sudoku(base_board.copy(), base_solution.copy())

            # Convert to tensors
            x = puzzle_to_tensor(aug_board)
            holes, targets = get_targets_from_arr(aug_board, aug_solution)

            x_list.append(x)
            holes_list.append((holes, targets))
            counts_list.append(len(holes))

        x_batch = torch.stack(x_list).to(device)
        holes_batch = [(h[0].to(device), h[1].to(device)) for h in holes_list]
        return x_batch, holes_batch, counts_list

    def compute_loss(x_batch, holes_batch, counts_batch):
        logits = model(x_batch)
        hole_c = torch.cat([h[0] for h in holes_batch])
        targets = torch.cat([h[1] for h in holes_batch])
        counts = torch.tensor(counts_batch, device=device)
        hole_b = torch.repeat_interleave(torch.arange(len(holes_batch), device=device), counts)
        logits_holes = logits[hole_b, hole_c]
        loss = F.cross_entropy(logits_holes, targets)
        return loss, logits, hole_b, hole_c, targets

    def evaluate_all():
        model.eval()
        results = {}
        total_solved = 0
        total_puzzles = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for name, data in test_data.items():
                x_test = data['x']
                puzzles = data['puzzles']
                solutions = data['solutions']
                puzzles_solved = 0
                for start in range(0, len(puzzles), 256):
                    end = min(start + 256, len(puzzles))
                    batch_x = x_test[start:end]
                    logits = model(batch_x)
                    preds_full = logits.argmax(dim=-1).cpu()
                    for b, (puzzle, solution) in enumerate(zip(puzzles[start:end], solutions[start:end])):
                        pred_solution = list(puzzle)
                        for i in range(81):
                            if puzzle[i] == '.':
                                pred_solution[i] = str(preds_full[b, i].item() + 1)
                        if ''.join(pred_solution) == solution:
                            puzzles_solved += 1
                results[name] = {'solved': puzzles_solved, 'total': len(puzzles)}
                total_solved += puzzles_solved
                total_puzzles += len(puzzles)
        results['_total'] = {'solved': total_solved, 'total': total_puzzles}
        return results

    def do_save_checkpoint(step):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt")
        save_checkpoint(path, step, model, optimizer, CONFIG)
        print(f"Checkpoint saved: {path}")

    for step in range(start_step, total_steps):
        model.train()
        x_batch, holes_batch, counts_batch = sample_augmented_batch(batch_size)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logits, hole_b, hole_c, targets = compute_loss(x_batch, holes_batch, counts_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                preds = logits[hole_b, hole_c].argmax(dim=-1)
                train_acc = (preds == targets).float().mean().item()

            if step % 5000 == 0 or step == total_steps - 1:
                results = evaluate_all()
                total_r = results.pop('_total')
                log(f"Step {step:5d} | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - TRM with Augmentation")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    log(f"Baseline (BS=4096): 76.3%")
    log(f"TRM: 87.4%")

    final_path = os.path.join(output_dir, "model_trm_augment.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()


if __name__ == "__main__":
    train()
