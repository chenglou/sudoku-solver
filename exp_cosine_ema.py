# Experiment: EMA on top of cosine LR (retest)
# Based on exp_cosine_no_sam.py (83.6% SOTA)
# Changes:
#   - Add EMA with decay=0.999, evaluate with shadow weights
# Hypothesis: EMA was tested on warmup baseline (hurt -1pp). Maybe it helps with cosine?
# TRM gets +7.5pp from EMA, but we don't know if they had cosine too.

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import os
import math
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, restore_optimizer, save_checkpoint

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "cosine_ema_checkpoint_step"

CONFIG = {
    'experiment': 'exp_cosine_ema',
    'd_model': 128,
    'n_layers': 4,
    'batch_size': 4096,
    'lr': 1.5e-3,
    'warmup_steps': 2000,
    'lr_min_ratio': 0.01,
    'ema_decay': 0.999,
}

d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16
lr = 1.5e-3
warmup_steps = 2000
lr_min_ratio = 0.01
total_steps = 70000
batch_size = 4096
train_size = 2700000
ema_decay = 0.999

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


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model weights with shadow weights."""
        self.backup = {name: param.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


class SudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        self.row_embed = nn.Embedding(9, d_model)
        self.col_embed = nn.Embedding(9, d_model)
        self.box_embed = nn.Embedding(9, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 9)

    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        device = x.device
        row_idx = ROW_IDX.to(device)
        col_idx = COL_IDX.to(device)
        box_idx = BOX_IDX.to(device)
        pos_embed = self.row_embed(row_idx) + self.col_embed(col_idx) + self.box_embed(box_idx)

        h_prev = self.initial_encoder(x)
        preds = torch.zeros(batch_size, 81, 9, device=device)

        all_logits = []
        for _ in range(n_iterations):
            h = h_prev + self.pred_proj(preds) + pos_embed
            h = self.transformer(h)
            h_prev = h
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
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


def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (lr_min_ratio + (1 - lr_min_ratio) * cosine_decay)


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

    model = SudokuTransformer().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    checkpoint_path, start_step = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    checkpoint_data = None
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    # Initialize EMA before compile
    ema = EMA(model, decay=ema_decay)

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed from step {start_step}")

    print(f"\nExperiment: Cosine LR with EMA (decay={ema_decay})")
    print(f"Architecture: d_model={d_model}, n_layers={n_layers}")
    print(f"Batch size: {batch_size}, lr: {lr}")
    print(f"Output directory: {output_dir}")

    log_path = os.path.join(output_dir, "exp_cosine_ema.log")
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
        x_list, holes_list, counts_list = [], [], []
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
        all_logits = model(x_batch, return_all=True)
        hole_c = torch.cat([h[0] for h in holes_batch])
        targets = torch.cat([h[1] for h in holes_batch])
        counts = torch.tensor(counts_batch, device=device)
        hole_b = torch.repeat_interleave(torch.arange(len(holes_batch), device=device), counts)
        loss = 0
        for logits in all_logits:
            logits_holes = logits[hole_b, hole_c]
            loss = loss + F.cross_entropy(logits_holes, targets)
        loss = loss / len(all_logits)
        return loss, all_logits, hole_b, hole_c, targets

    def evaluate_all(use_ema=True):
        model.eval()
        if use_ema:
            ema.apply(model)
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
        if use_ema:
            ema.restore(model)
        results['_total'] = {'solved': total_solved, 'total': total_puzzles}
        return results

    def do_save_checkpoint(step):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow,
            'config': CONFIG,
        }, path)
        print(f"Checkpoint saved: {path}")

    current_phase_name = None
    current_buckets = None

    for step in range(start_step, total_steps):
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

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
            loss, all_logits, hole_b, hole_c, targets = compute_loss(x_batch, holes_batch, counts_batch)
        loss.backward()
        optimizer.step()

        # Update EMA after each step
        ema.update(model)

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                final_logits_holes = all_logits[-1][hole_b, hole_c]
                preds = final_logits_holes.argmax(dim=-1)
                train_acc = (preds == targets).float().mean().item()

            if step % 5000 == 0 or step == total_steps - 1:
                results = evaluate_all(use_ema=True)
                total_r = results.pop('_total')
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - Cosine LR with EMA")
    log("="*60)
    results = evaluate_all(use_ema=True)
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal (EMA): {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")

    # Also evaluate without EMA for comparison
    results_no_ema = evaluate_all(use_ema=False)
    total_no_ema = results_no_ema.pop('_total')
    log(f"Total (no EMA): {total_no_ema['solved']}/{total_no_ema['total']} ({100*total_no_ema['solved']/total_no_ema['total']:.1f}%)")
    log(f"Baseline (no EMA): 83.6%")

    final_path = os.path.join(output_dir, "model_cosine_ema.pt")
    state_dict = ema.shadow  # Save EMA weights
    torch.save(state_dict, final_path)
    log(f"Final model saved (EMA weights): {final_path}")
    log_file.close()


if __name__ == "__main__":
    train()
