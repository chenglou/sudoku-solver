# Experiment: EXACT TRM reproduction v2
#
# Key fix from v1: TRM trains for 50K EPOCHS over 1M examples, not 50K steps!
# - 1K puzzles × 1000 augmentations = 1M pre-generated examples
# - 50K epochs × (1M / BS) steps per epoch = ~65M steps total
# - They pre-generate augmentations (not on-the-fly)
#
# TRM settings:
#   - 1000 random puzzles, 1000 augmentations each = 1M examples
#   - BS=768, lr=1e-4, weight_decay=1.0
#   - MLP-T (no attention), no position encoding
#   - H_cycles=3, L_cycles=6, L_layers=2, hidden_size=512
#   - EMA with decay 0.999
#   - 50K epochs (~65M steps, ~20 hours on L40S)

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import numpy as np
import os
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, restore_optimizer, save_checkpoint
from tensorboard_utils import TBLogger

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "trm_exact_v2_checkpoint_step"

# EXACT TRM hyperparameters
hidden_size = 512
L_layers = 2
H_cycles = 3
L_cycles = 6
expansion = 4
lr = 1e-4
weight_decay = 1.0
batch_size = 768  # TRM uses 768
ema_decay = 0.999

# EXACT TRM data settings
n_base_puzzles = 1000
n_augmentations = 1000  # Per puzzle
n_epochs = 50000  # TRM uses 50K epochs

CONFIG = {
    'experiment': 'exp_trm_exact_v2',
    'hidden_size': hidden_size,
    'L_layers': L_layers,
    'H_cycles': H_cycles,
    'L_cycles': L_cycles,
    'batch_size': batch_size,
    'lr': lr,
    'n_base_puzzles': n_base_puzzles,
    'n_augmentations': n_augmentations,
    'n_epochs': n_epochs,
}

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """TRM-style sudoku augmentation."""
    digit_map = np.concatenate([[0], np.random.permutation(np.arange(1, 10))])
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T.copy()
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
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


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name.replace('_orig_mod.', '')
                self.shadow[clean_name] = param.data.clone()

    def _clean_name(self, name):
        return name.replace('_orig_mod.', '')

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = self._clean_name(name)
                self.shadow[clean_name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = self._clean_name(name)
                self.backup[clean_name] = param.data.clone()
                param.data.copy_(self.shadow[clean_name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = self._clean_name(name)
                param.data.copy_(self.backup[clean_name])
        self.backup = {}


def parse_puzzle(puzzle_str):
    arr = np.zeros((9, 9), dtype=np.int64)
    for i, c in enumerate(puzzle_str):
        if c != '.' and c != '0':
            arr[i // 9, i % 9] = int(c)
    return arr


def train(output_dir="."):
    device = torch.device("cuda")

    print("Loading sudoku-extreme train split...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    print(f"Total available: {len(dataset)}")

    # Sample base puzzles
    print(f"\nSampling {n_base_puzzles} RANDOM base puzzles...")
    all_indices = list(range(len(dataset)))
    selected_indices = random.sample(all_indices, n_base_puzzles)

    base_puzzles = []
    for idx in selected_indices:
        puzzle_str = dataset[idx]['question']
        solution_str = dataset[idx]['answer']
        board = parse_puzzle(puzzle_str)
        solution = parse_puzzle(solution_str)
        base_puzzles.append((board, solution))

    # Pre-generate ALL augmentations (like TRM does)
    print(f"Pre-generating {n_base_puzzles} × {n_augmentations} = {n_base_puzzles * n_augmentations} augmented examples...")
    all_examples = []
    for i, (board, solution) in enumerate(base_puzzles):
        # Include original
        all_examples.append((board.copy(), solution.copy()))
        # Add augmentations
        for _ in range(n_augmentations):
            aug_board, aug_solution = shuffle_sudoku(board.copy(), solution.copy())
            all_examples.append((aug_board, aug_solution))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_base_puzzles} puzzles processed...")

    n_examples = len(all_examples)
    print(f"Total training examples: {n_examples}")

    # Pre-encode all examples
    print("Encoding all examples...")
    all_x = torch.zeros(n_examples, 81, dtype=torch.long)
    all_holes = []
    all_targets = []

    for i, (board, solution) in enumerate(all_examples):
        all_x[i] = torch.tensor(board.flatten(), dtype=torch.long)
        board_flat = board.flatten()
        solution_flat = solution.flatten()
        holes = []
        targets = []
        for j in range(81):
            if board_flat[j] == 0:
                holes.append(j)
                targets.append(solution_flat[j] - 1)
        all_holes.append(torch.tensor(holes, dtype=torch.long))
        all_targets.append(torch.tensor(targets, dtype=torch.long))

    print("Examples encoded.")

    # Test data
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

    # Calculate training stats
    steps_per_epoch = (n_examples + batch_size - 1) // batch_size
    total_steps = n_epochs * steps_per_epoch
    print(f"\nTraining setup:")
    print(f"  Examples: {n_examples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Total steps: {total_steps:,}")

    checkpoint_path, start_step = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    checkpoint_data = None
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    ema = EMA(model, decay=ema_decay)
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

    print(f"\nExperiment: EXACT TRM Reproduction v2")
    print(f"Architecture: hidden_size={hidden_size}, L_layers={L_layers}, MLP-T")
    print(f"Iterations: H_cycles={H_cycles} × L_cycles={L_cycles}")
    print(f"Data: {n_base_puzzles} puzzles × {n_augmentations} augmentations (pre-generated)")
    print(f"Batch size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}")
    print(f"EMA decay: {ema_decay}")
    print(f"Output directory: {output_dir}")

    tb_logger = TBLogger(output_dir, "exp_trm_exact_v2")
    log_path = os.path.join(output_dir, "exp_trm_exact_v2.log")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def evaluate_all():
        ema.apply(model)
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
                            if puzzle[i] == '.' or puzzle[i] == '0':
                                pred_solution[i] = str(preds_full[b, i].item() + 1)
                        if ''.join(pred_solution) == solution:
                            puzzles_solved += 1
                results[name] = {'solved': puzzles_solved, 'total': len(puzzles)}
                total_solved += puzzles_solved
                total_puzzles += len(puzzles)
        results['_total'] = {'solved': total_solved, 'total': total_puzzles}
        ema.restore(model)
        return results

    def do_save_checkpoint(step):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt")
        save_checkpoint(path, step, model, optimizer, CONFIG)
        print(f"Checkpoint saved: {path}")

    # Training loop - epoch-based like TRM
    step = start_step
    start_epoch = start_step // steps_per_epoch

    for epoch in range(start_epoch, n_epochs):
        # Shuffle examples each epoch
        perm = torch.randperm(n_examples)

        model.train()
        for batch_start in range(0, n_examples, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]
            x_batch = all_x[batch_idx].to(device)

            # Gather holes and targets for batch
            holes_batch = [all_holes[i].to(device) for i in batch_idx]
            targets_batch = [all_targets[i].to(device) for i in batch_idx]
            counts = [len(h) for h in holes_batch]

            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x_batch)
                hole_c = torch.cat(holes_batch)
                targets = torch.cat(targets_batch)
                counts_t = torch.tensor(counts, device=device)
                hole_b = torch.repeat_interleave(torch.arange(len(holes_batch), device=device), counts_t)
                logits_holes = logits[hole_b, hole_c]
                loss = F.cross_entropy(logits_holes, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            step += 1

            # Logging
            if step % 1000 == 0:
                with torch.no_grad():
                    preds = logits_holes.argmax(dim=-1)
                    train_acc = (preds == targets).float().mean().item()

                tb_logger.log(step, loss=loss.item(), train_acc=train_acc * 100)

                if step % 50000 == 0:
                    results = evaluate_all()
                    total_r = results.pop('_total')
                    log(f"Step {step:7d} (epoch {epoch}) | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                        " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                        f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                    tb_logger.log(step, test_acc=100 * total_r['solved'] / total_r['total'])
                    for name, r in results.items():
                        tb_logger.log(step, **{f"Test/{name}": 100 * r['solved'] / r['total']})
                    do_save_checkpoint(step)
                else:
                    log(f"Step {step:7d} (epoch {epoch}) | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - EXACT TRM Reproduction v2")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    log(f"Our baseline: 76.3%")
    log(f"TRM claimed: 87.4%")

    final_path = os.path.join(output_dir, "model_trm_exact_v2.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    tb_logger.close()


if __name__ == "__main__":
    train()
