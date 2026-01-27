# Experiment: Nested loops on our transformer (minimal change)
# Tests if nested H_cycles/L_cycles structure helps WITHOUT changing architecture
#
# Key change: Instead of 16 flat iterations, use H_cycles=3 × L_cycles=6 = 18 iterations
# with first (H_cycles-1) outer loops having no gradients
#
# This isolates the nested loop effect from architecture changes (MLP-T, hidden_size=512, etc.)

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import os
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, restore_optimizer, save_checkpoint
from tensorboard_utils import TBLogger

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "nested_loops_checkpoint_step"

# Keep our baseline architecture, just change iteration structure
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4

# TRM-style iteration structure
H_cycles = 3       # Outer loops
L_cycles = 6       # Inner loops per outer loop

lr = 1e-4          # TRM uses 1e-4 (our 1.5e-3 caused collapse)
total_steps = 70000
batch_size = 4096  # Keep our best batch size
train_size = 2700000

CONFIG = {
    'experiment': 'exp_nested_loops',
    'd_model': d_model,
    'n_layers': n_layers,
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

ROW_IDX = torch.tensor([i // 9 for i in range(81)])
COL_IDX = torch.tensor([i % 9 for i in range(81)])
BOX_IDX = torch.tensor([(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)])


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (from baseline)."""
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho)
        params = list(params)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = group.get('rho', 0.05)
            scale = rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_w'] = p.data.clone()
                p.data.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]['old_w'])
        self.base_optimizer.step()

    @torch.no_grad()
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([p.grad.norm() for group in self.param_groups for p in group['params'] if p.grad is not None])
        )
        return norm

    def zero_grad(self):
        self.base_optimizer.zero_grad()


class SudokuTransformerNested(nn.Module):
    """
    Our baseline transformer but with TRM-style nested iteration.

    Key differences from baseline:
    1. Two latent states: z_H (high-level) and z_L (low-level)
    2. Nested loops: H_cycles × L_cycles
    3. First (H_cycles-1) outer loops have no gradients
    """
    def __init__(self, H_cycles, L_cycles):
        super().__init__()
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Same input encoding as baseline
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        self.row_embed = nn.Embedding(9, d_model)
        self.col_embed = nn.Embedding(9, d_model)
        self.box_embed = nn.Embedding(9, d_model)

        # Same transformer as baseline
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_head = nn.Linear(d_model, 9)

        # Initial states for z_H and z_L
        self.H_init = nn.Parameter(torch.randn(d_model) * 0.02)
        self.L_init = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        device = x.device
        row_idx = ROW_IDX.to(device)
        col_idx = COL_IDX.to(device)
        box_idx = BOX_IDX.to(device)
        pos_embed = self.row_embed(row_idx) + self.col_embed(col_idx) + self.box_embed(box_idx)

        # Input embedding
        input_embed = self.initial_encoder(x)  # (B, 81, D)

        # Initialize latent states
        z_H = self.H_init.expand(batch_size, 81, -1)  # (B, 81, D)
        z_L = self.L_init.expand(batch_size, 81, -1)  # (B, 81, D)
        preds = torch.zeros(batch_size, 81, 9, device=device)

        all_logits = []

        # First (H_cycles-1) outer loops WITHOUT gradients
        with torch.no_grad():
            for _H_step in range(self.H_cycles - 1):
                # Inner L_cycles: update z_L using z_H + input
                for _L_step in range(self.L_cycles):
                    h = z_L + self.pred_proj(preds) + z_H + input_embed + pos_embed
                    h = self.transformer(h)
                    z_L = h
                    logits = self.output_head(h)
                    preds = F.softmax(logits, dim=-1)

                # Update z_H at end of each outer cycle
                h = z_H + z_L + input_embed + pos_embed
                h = self.transformer(h)
                z_H = h

                if return_all:
                    all_logits.append(logits.detach())

        # Last H_cycle WITH gradients
        for _L_step in range(self.L_cycles):
            h = z_L + self.pred_proj(preds) + z_H + input_embed + pos_embed
            h = self.transformer(h)
            z_L = h
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
            if return_all:
                all_logits.append(logits)

        # Final z_H update
        h = z_H + z_L + input_embed + pos_embed
        h = self.transformer(h)
        z_H = h
        logits = self.output_head(z_H)

        if return_all:
            all_logits.append(logits)

        return all_logits if return_all else logits


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

    model = SudokuTransformerNested(H_cycles=H_cycles, L_cycles=L_cycles).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,} (baseline: ~800K)")

    checkpoint_path, start_step = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    checkpoint_data = None
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    model = torch.compile(model)

    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=lr, betas=(0.9, 0.95))

    if checkpoint_data:
        restore_optimizer(optimizer, checkpoint_data, device)
        print(f"Resumed from step {start_step}")

    print(f"\nExperiment: Nested Loops on Transformer")
    print(f"Architecture: d_model={d_model}, n_layers={n_layers} (same as baseline)")
    print(f"Iterations: H_cycles={H_cycles} × L_cycles={L_cycles} (first {H_cycles-1} no grad)")
    print(f"Batch size: {batch_size}, lr: {lr}")
    print(f"Output directory: {output_dir}")

    # TensorBoard logger
    tb_logger = TBLogger(output_dir, "exp_nested_loops")

    log_path = os.path.join(output_dir, "exp_nested_loops.log")
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
        # Only use final output (TRM style: no intermediate supervision)
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
            loss1, _, _, _, _ = compute_loss(x_batch, holes_batch, counts_batch)
        loss1.backward()
        optimizer.first_step()

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss2, logits, hole_b, hole_c, targets = compute_loss(x_batch, holes_batch, counts_batch)
        loss2.backward()
        optimizer.second_step()

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                preds = logits[hole_b, hole_c].argmax(dim=-1)
                train_acc = (preds == targets).float().mean().item()

            # Log to TensorBoard
            tb_logger.log(step, loss=loss2.item(), train_acc=train_acc * 100)

            if step % 5000 == 0 or step == total_steps - 1:
                results = evaluate_all()
                total_r = results.pop('_total')
                log(f"Step {step:5d} | Loss: {loss2.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                # Log test metrics to TensorBoard
                tb_logger.log(step, test_acc=100 * total_r['solved'] / total_r['total'])
                for name, r in results.items():
                    tb_logger.log(step, **{f"Test/{name}": 100 * r['solved'] / r['total']})
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | Loss: {loss2.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - Nested Loops")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    log(f"Baseline (16 flat iterations): 76.3%")
    log(f"TRM: 87.4%")

    final_path = os.path.join(output_dir, "model_nested_loops.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    tb_logger.close()


if __name__ == "__main__":
    train()
