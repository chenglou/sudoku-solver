# Experiment: Sudoku-agnostic positional embedding
# Based on exp_faster.py but replaces row/col/box embeddings with learned absolute positions.

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import random
import os
import math
import numpy as np
from checkpoint_utils import find_latest_checkpoint, load_checkpoint

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "faster_abspos_checkpoint_step"

CONFIG = {
    'experiment': 'exp_faster_abspos',
    'd_model': 128,
    'd_ff': 512,
    'n_layers': 4,
    'batch_size': 4096,
    'lr': 1.5e-3,
    'warmup_steps': 1400,
    'lr_min_ratio': 0.01,
    'total_steps': 50000,
}

d_model = 128  # head_dim=32 (d_model / n_heads)
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16
lr = 1.5e-3
warmup_steps = 1400
lr_min_ratio = 0.01
total_steps = 50000
batch_size = 4096
train_size = 2700000
eval_every = 5000
checkpoint_prefix = CHECKPOINT_PREFIX
log_name = "exp_faster_abspos.log"

# Same curriculum as 50K baseline
PHASES = [
    (0, 10000, 21, "Phase 1: Hard only (rating 21+)"),
    (10000, 20000, 6, "Phase 2: Medium+ (rating 6+)"),
    (20000, 30000, 1, "Phase 3: Easy+ (rating 1+)"),
    (30000, 50000, 0, "Phase 4: All (rating 0+)"),
]

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]

POS_IDX = torch.arange(81)

CHAR_TO_DIGIT = np.zeros(256, dtype=np.uint8)
for i in range(1, 10):
    CHAR_TO_DIGIT[ord(str(i))] = i

CHAR_TO_TARGET = np.zeros(256, dtype=np.uint8)
for i in range(1, 10):
    CHAR_TO_TARGET[ord(str(i))] = i - 1

CHAR_TO_DIGIT[ord('.')] = 0

ENCODE_CHUNK_SIZE = 50000
ONE_HOT = np.eye(10, dtype=np.float32)


class SudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        # Sudoku-agnostic learned absolute position embedding
        self.pos_embed = nn.Embedding(81, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 9)

    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        device = x.device
        pos_embed = self.pos_embed(POS_IDX.to(device))

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


def encode_puzzles(puzzles):
    if not puzzles:
        return torch.empty((0, 81, 10), dtype=torch.float32)
    chunks = []
    for start in range(0, len(puzzles), ENCODE_CHUNK_SIZE):
        chunk = puzzles[start:start + ENCODE_CHUNK_SIZE]
        buf = ''.join(chunk).encode('ascii')
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(chunk), 81)
        digits = CHAR_TO_DIGIT[arr]
        chunks.append(torch.from_numpy(ONE_HOT[digits]))
    return torch.cat(chunks, dim=0)


def encode_solutions(solutions):
    if not solutions:
        return torch.empty((0, 81), dtype=torch.uint8)
    chunks = []
    for start in range(0, len(solutions), ENCODE_CHUNK_SIZE):
        chunk = solutions[start:start + ENCODE_CHUNK_SIZE]
        buf = ''.join(chunk).encode('ascii')
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(chunk), 81)
        digits = CHAR_TO_TARGET[arr]
        chunks.append(torch.from_numpy(digits))
    return torch.cat(chunks, dim=0)


def get_lr(step):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (lr_min_ratio + (1 - lr_min_ratio) * cosine_decay)


def train(output_dir="."):
    device = torch.device("cuda")
    print(
        "SDPA backends enabled: "
        f"flash={torch.backends.cuda.flash_sdp_enabled()}, "
        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
        f"math={torch.backends.cuda.math_sdp_enabled()}"
    )

    print("Loading sudoku-extreme train split...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    print(f"Total available: {len(dataset)}")
    train_size_local = min(train_size, len(dataset))
    print(f"Using first {train_size_local} for training")

    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    print(f"Test set: {len(test_dataset)}")

    print("\nEncoding training data by rating...")
    train_rows = dataset[:train_size_local]
    ratings = np.asarray(train_rows["rating"], dtype=np.int16)
    puzzles_all = train_rows["question"]
    solutions_all = train_rows["answer"]
    x_all = encode_puzzles(puzzles_all)
    targets_all = encode_solutions(solutions_all)
    del puzzles_all, solutions_all, train_rows
    train_data = {}
    for min_r, max_r, name in RATING_BUCKETS:
        idx = np.where((ratings >= min_r) & (ratings <= max_r))[0]
        if idx.size == 0:
            continue
        print(f"  Rating {name}: {len(idx)} puzzles...", end=" ", flush=True)
        train_data[(min_r, max_r)] = {
            'idx': torch.from_numpy(idx),
            'size': int(idx.size),
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
        x_test = encode_puzzles(puzzles).to(device)
        test_data[name] = {
            'x': x_test,
            'puzzles': puzzles,
            'solutions': solutions,
        }
        print(f"  Test {name}: {len(puzzles)} puzzles")

    model = SudokuTransformer().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    checkpoint_path, start_step = None, 0
    checkpoint_data = None
    checkpoint_path, start_step = find_latest_checkpoint(output_dir, checkpoint_prefix)
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Resumed from step {start_step}")

    print(f"\nExperiment: Abs pos (d_model={d_model}, head_dim={d_model // n_heads})")
    print(f"Architecture: d_model={d_model}, d_ff={d_ff}, n_layers={n_layers}")
    print(f"Batch size: {batch_size}, lr: {lr}, warmup_steps: {warmup_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Evaluation interval: {eval_every}")
    print(f"Output directory: {output_dir}")

    log_path = os.path.join(output_dir, log_name)
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
        sizes = np.array([train_data[b]['size'] for b in active_buckets], dtype=np.int64)
        total = sizes.sum()
        probs = sizes / total
        counts = np.random.multinomial(bs, probs)
        x_parts = []
        t_parts = []
        for b, count in zip(active_buckets, counts):
            if count == 0:
                continue
            bucket_idx = train_data[b]['idx']
            sel = bucket_idx[torch.randint(0, train_data[b]['size'], (count,))]
            x_parts.append(x_all[sel])
            t_parts.append(targets_all[sel])
        x_batch = torch.cat(x_parts, dim=0)
        t_batch = torch.cat(t_parts, dim=0)
        perm = torch.randperm(bs)
        x_batch = x_batch[perm]
        t_batch = t_batch[perm]
        return x_batch, t_batch

    def compute_loss(x_batch, t_batch):
        all_logits = model(x_batch, return_all=True)
        mask = x_batch[:, :, 0]
        mask = mask.to(dtype=torch.float32)
        t_batch = t_batch.to(dtype=torch.long)
        loss = 0
        for logits in all_logits:
            per_cell = F.cross_entropy(logits.reshape(-1, 9), t_batch.reshape(-1), reduction='none')
            per_cell = per_cell.view(t_batch.size(0), 81)
            loss = loss + (per_cell * mask).sum() / mask.sum()
        loss = loss / len(all_logits)
        return loss, all_logits, mask, t_batch

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
        path = os.path.join(output_dir, f"{checkpoint_prefix}{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
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
        x_batch, t_batch = sample_batch(current_buckets, batch_size)
        x_batch = x_batch.to(device)
        t_batch = t_batch.to(device)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, all_logits, mask, t_batch = compute_loss(x_batch, t_batch)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                final_logits = all_logits[-1]
                preds = final_logits.argmax(dim=-1)
                correct = (preds == t_batch) & (mask > 0)
                train_acc = correct.sum().item() / mask.sum().item()

            do_eval = step % eval_every == 0 or step == total_steps - 1
            if do_eval:
                results = evaluate_all()
                total_r = results.pop('_total')
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - exp_faster_abspos")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    log(f"Cosine 50K baseline: 82.8%")

    final_path = os.path.join(output_dir, "model_faster_abspos.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()


if __name__ == "__main__":
    train()
