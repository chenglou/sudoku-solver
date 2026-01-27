# Experiment: Reproduce TRM's nested H_cycles/L_cycles structure
# TRM achieves 87.4% on sudoku-extreme with:
#   - H_cycles=3 (outer loops), L_cycles=6 (inner loops)
#   - First (H_cycles-1) outer loops WITHOUT gradients
#   - Two latent states: z_H (high-level) and z_L (low-level)
#   - hidden_size=512, L_layers=2
#   - MLP-T (sequence-wise MLP instead of attention, no position encoding)
#   - 1K examples × 1000 digit relabelings
#
# Our baseline: 76.3% with single-level 16 iterations, 128 dim, 4 layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import os
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, restore_optimizer, save_checkpoint
from tensorboard_utils import TBLogger

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "trm_nested_checkpoint_step"

# TRM-style hyperparameters
hidden_size = 512  # TRM uses 512
L_layers = 2       # TRM uses 2 blocks per reasoning module
H_cycles = 3       # Outer loops
L_cycles = 6       # Inner loops per outer loop
expansion = 4      # MLP expansion factor (TRM uses 4)
lr = 1e-4          # TRM uses 1e-4 (we were using 1.5e-3)
weight_decay = 1.0 # TRM uses 1.0
total_steps = 70000
batch_size = 4096  # Keep our best batch size (TRM uses 256)
train_size = 2700000

CONFIG = {
    'experiment': 'exp_trm_nested',
    'hidden_size': hidden_size,
    'L_layers': L_layers,
    'H_cycles': H_cycles,
    'L_cycles': L_cycles,
    'batch_size': batch_size,
    'lr': lr,
}

# Reverse curriculum phases
PHASES = [
    (0, 14000, 21, "Phase 1: Hard only (rating 21+)"),
    (14000, 28000, 6, "Phase 2: Medium+ (rating 6+)"),
    (28000, 42000, 1, "Phase 3: Easy+ (rating 1+)"),
    (42000, 70000, 0, "Phase 4: All (rating 0+)"),
]

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def rms_norm(x, eps=1e-5):
    """RMS normalization like TRM uses."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class SwiGLU(nn.Module):
    """SwiGLU MLP like TRM uses."""
    def __init__(self, hidden_size, expansion=4):
        super().__init__()
        inner_size = int(hidden_size * expansion * 2 / 3)  # SwiGLU standard
        inner_size = (inner_size + 63) // 64 * 64  # Round to 64 for efficiency
        self.w1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.w2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.w3 = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TRMBlock(nn.Module):
    """A single TRM-style block: MLP-T + channel MLP."""
    def __init__(self, hidden_size, seq_len=81, expansion=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # MLP-T: sequence-wise MLP (instead of attention)
        # Operates on transposed input: (B, D, L) -> (B, D, L)
        self.mlp_t = SwiGLU(seq_len, expansion)

        # Channel MLP (standard FFN)
        self.mlp = SwiGLU(hidden_size, expansion)

    def forward(self, x):
        # x: (B, L, D)

        # MLP-T: mix across sequence positions
        x_t = x.transpose(1, 2)  # (B, D, L)
        x_t = self.mlp_t(x_t)
        x = rms_norm(x + x_t.transpose(1, 2))  # Post-norm

        # Channel MLP
        x = rms_norm(x + self.mlp(x))

        return x


class ReasoningModule(nn.Module):
    """TRM's L_level: stack of L_layers blocks."""
    def __init__(self, hidden_size, L_layers, seq_len=81, expansion=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, seq_len, expansion)
            for _ in range(L_layers)
        ])

    def forward(self, hidden_states, input_injection):
        """
        Args:
            hidden_states: current latent state (z_L or z_H)
            input_injection: what to add before processing (z_H + embeddings or z_L)
        """
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TRMSudoku(nn.Module):
    """
    TRM-style model for Sudoku with nested H_cycles/L_cycles.

    Key differences from our baseline:
    1. Two latent states: z_H (high-level) and z_L (low-level)
    2. Nested loops: H_cycles outer × L_cycles inner
    3. First (H_cycles-1) outer loops have no gradients
    4. MLP-T instead of attention
    5. No position encoding (Sudoku has fixed structure)
    """
    def __init__(self, hidden_size, L_layers, H_cycles, L_cycles, expansion=4, seq_len=81):
        super().__init__()
        self.hidden_size = hidden_size
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Input embedding: 10 classes (0=empty, 1-9=digits)
        self.embed_tokens = nn.Embedding(10, hidden_size)
        self.embed_scale = hidden_size ** 0.5

        # Reasoning module (shared across all iterations like TRM)
        self.L_level = ReasoningModule(hidden_size, L_layers, seq_len, expansion)

        # Initial states for z_H and z_L (learned, like TRM)
        self.H_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(hidden_size) * 0.02)

        # Output head
        self.lm_head = nn.Linear(hidden_size, 9, bias=False)

    def forward(self, x, return_all=False):
        """
        Args:
            x: input puzzle (B, 81) with values 0-9 (0=empty)
        Returns:
            logits: (B, 81, 9) predictions
        """
        batch_size = x.size(0)
        device = x.device

        # Input embedding
        input_embeddings = self.embed_tokens(x) * self.embed_scale  # (B, 81, D)

        # Initialize latent states
        z_H = self.H_init.expand(batch_size, 81, -1)  # (B, 81, D)
        z_L = self.L_init.expand(batch_size, 81, -1)  # (B, 81, D)

        all_logits = []

        # First (H_cycles-1) outer loops WITHOUT gradients
        with torch.no_grad():
            for _H_step in range(self.H_cycles - 1):
                # Inner L_cycles: update z_L
                for _L_step in range(self.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings)
                # Update z_H at end of each outer cycle
                z_H = self.L_level(z_H, z_L)

                if return_all:
                    logits = self.lm_head(z_H)
                    all_logits.append(logits.detach())

        # Last H_cycle WITH gradients
        for _L_step in range(self.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings)
        z_H = self.L_level(z_H, z_L)

        logits = self.lm_head(z_H)

        if return_all:
            all_logits.append(logits)
            return all_logits
        return logits


def encode_puzzle(puzzle_str):
    """Convert puzzle string to tensor (0 for empty, 1-9 for digits)."""
    x = torch.zeros(81, dtype=torch.long)
    for i, c in enumerate(puzzle_str):
        if c == '.' or c == '0':
            x[i] = 0
        else:
            x[i] = int(c)
    return x


def get_targets(puzzle_str, solution_str):
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.' or p == '0':
            holes.append(i)
            targets.append(int(s) - 1)  # 0-indexed for cross entropy
    return torch.tensor(holes), torch.tensor(targets)


def train(output_dir="."):
    device = torch.device("cuda")

    print("Loading sudoku-extreme train split...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    print(f"Total available: {len(dataset)}")
    print(f"Using first {train_size} for training")

    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    print(f"Test set: {len(test_dataset)}")

    print("\nEncoding training data by rating...")
    train_data = {}
    for min_r, max_r, name in RATING_BUCKETS:
        indices = [i for i in range(train_size) if min_r <= dataset[i]['rating'] <= max_r]
        if len(indices) == 0:
            continue
        print(f"  Rating {name}: {len(indices)} puzzles...", end=" ", flush=True)
        puzzles = [dataset[i]['question'] for i in indices]
        solutions = [dataset[i]['answer'] for i in indices]
        x_data = torch.stack([encode_puzzle(p) for p in puzzles])
        holes_list = [get_targets(p, s) for p, s in zip(puzzles, solutions)]
        holes_count = [len(h[0]) for h in holes_list]
        train_data[(min_r, max_r)] = {
            'x': x_data,
            'holes': holes_list,
            'holes_count': holes_count,
            'size': len(puzzles),
        }
        print("done")

    phase_buckets = {}
    for start, end, min_rating, name in PHASES:
        buckets_for_phase = [k for k in train_data.keys() if k[0] >= min_rating]
        total = sum(train_data[k]['size'] for k in buckets_for_phase)
        phase_buckets[min_rating] = buckets_for_phase
        print(f"  {name}: {total} puzzles")

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
        x_test = torch.stack([encode_puzzle(p) for p in puzzles]).to(device)
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
    print(f"\nModel parameters: {param_count:,} (TRM: ~5M, our baseline: ~800K)")

    # Check for checkpoint BEFORE torch.compile()
    checkpoint_path, start_step = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    checkpoint_data = None
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    model = torch.compile(model)

    # TRM-style optimizer: AdamW with high weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),  # TRM uses these betas
        weight_decay=weight_decay,
    )

    if checkpoint_data:
        restore_optimizer(optimizer, checkpoint_data, device)
        print(f"Resumed from step {start_step}")

    print(f"\nExperiment: TRM Nested H_cycles/L_cycles")
    print(f"Architecture: hidden_size={hidden_size}, L_layers={L_layers}")
    print(f"Iterations: H_cycles={H_cycles} × L_cycles={L_cycles} (first {H_cycles-1} no grad)")
    print(f"Batch size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}")
    print(f"Output directory: {output_dir}")

    # TensorBoard logger
    tb_logger = TBLogger(output_dir, "exp_trm_nested")

    log_path = os.path.join(output_dir, "exp_trm_nested.log")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def get_phase(step):
        for start, end, min_rating, name in PHASES:
            if start <= step < end:
                return phase_buckets[min_rating], name
        return None, None

    def sample_batch(active_buckets, bs):
        sizes = [train_data[b]['size'] for b in active_buckets]
        total = sum(sizes)
        x_list = []
        holes_list = []
        counts_list = []
        for _ in range(bs):
            r = random.randint(0, total - 1)
            cumsum = 0
            for b, s in zip(active_buckets, sizes):
                cumsum += s
                if r < cumsum:
                    idx = random.randint(0, s - 1)
                    x_list.append(train_data[b]['x'][idx])
                    holes_list.append(train_data[b]['holes'][idx])
                    counts_list.append(train_data[b]['holes_count'][idx])
                    break
        x_batch = torch.stack(x_list).to(device)
        holes_batch = [(h[0].to(device), h[1].to(device)) for h in holes_list]
        return x_batch, holes_batch, counts_list

    def compute_loss(x_batch, holes_batch, counts_batch):
        logits = model(x_batch)  # Only final output (TRM style: no intermediate supervision)
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
                            if puzzle[i] == '.' or puzzle[i] == '0':
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

    current_phase_name = None
    current_buckets = None

    for step in range(start_step, total_steps):
        buckets, phase_name = get_phase(step)
        if phase_name != current_phase_name:
            current_phase_name = phase_name
            current_buckets = buckets
            total_puzzles = sum(train_data[b]['size'] for b in buckets)
            log(f"\n{'='*60}")
            log(f"Step {step}: Entering {phase_name}")
            log(f"Training pool: {total_puzzles} puzzles")
            log(f"{'='*60}\n")

        model.train()
        x_batch, holes_batch, counts_batch = sample_batch(current_buckets, batch_size)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logits, hole_b, hole_c, targets = compute_loss(x_batch, holes_batch, counts_batch)
        loss.backward()

        # Gradient clipping (TRM uses this implicitly via weight decay)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                preds = logits[hole_b, hole_c].argmax(dim=-1)
                train_acc = (preds == targets).float().mean().item()

            # Log to TensorBoard
            tb_logger.log(step, loss=loss.item(), train_acc=train_acc * 100)

            if step % 5000 == 0 or step == total_steps - 1:
                results = evaluate_all()
                total_r = results.pop('_total')
                log(f"Step {step:5d} | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                # Log test metrics to TensorBoard
                tb_logger.log(step, test_acc=100 * total_r['solved'] / total_r['total'])
                for name, r in results.items():
                    tb_logger.log(step, **{f"Test/{name}": 100 * r['solved'] / r['total']})
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - TRM Nested H_cycles/L_cycles")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    log(f"Baseline (BS=4096): 76.3%")
    log(f"TRM: 87.4%")

    final_path = os.path.join(output_dir, "model_trm_nested.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    tb_logger.close()


if __name__ == "__main__":
    train()
