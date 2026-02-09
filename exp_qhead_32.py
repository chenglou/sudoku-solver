# Experiment: Q-head with 32 training iterations
# Based on exp_qhead.py but trains with 32 iterations instead of 16.
# Hypothesis: training with 32 iters lets the model learn stable refinement
# over more steps (16-iter model gets 88.4% at 32 test iters but diverges at 48+).
# Eval: test at 32 iters (standard) and 48 iters (with Q-head early stop).

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

CHECKPOINT_PREFIX = "qhead32_checkpoint_step"

CONFIG = {
    'experiment': 'exp_qhead_32',
    'd_model': 128,
    'd_ff': 512,
    'n_layers': 4,
    'batch_size': 2048,
    'lr': 1.5e-3,
    'warmup_steps': 1400,
    'lr_min_ratio': 0.01,
    'total_steps': 50000,
    'n_iterations': 32,
    'q_loss_weight': 0.5,
}

d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 32
lr = 1.5e-3
warmup_steps = 1400
lr_min_ratio = 0.01
total_steps = 50000
batch_size = 2048
train_size = 2700000
eval_every = 5000
checkpoint_prefix = CHECKPOINT_PREFIX
log_name = "exp_qhead_32.log"
q_loss_weight = 0.5

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

ROW_IDX = torch.tensor([i // 9 for i in range(81)])
COL_IDX = torch.tensor([i % 9 for i in range(81)])

CHAR_TO_DIGIT = np.zeros(256, dtype=np.uint8)
for i in range(1, 10):
    CHAR_TO_DIGIT[ord(str(i))] = i

CHAR_TO_TARGET = np.zeros(256, dtype=np.uint8)
for i in range(1, 10):
    CHAR_TO_TARGET[ord(str(i))] = i - 1

CHAR_TO_DIGIT[ord('.')] = 0

ENCODE_CHUNK_SIZE = 50000
ONE_HOT = np.eye(10, dtype=np.float32)

# Precompute 2D RoPE cos/sin for all 81 positions
head_dim = d_model // n_heads  # 32
rope_half = head_dim // 2  # 16 dims per spatial axis
rope_pairs = rope_half // 2  # 8 frequency pairs per axis
rope_base = 10.0

_freqs = 1.0 / (rope_base ** (torch.arange(rope_pairs).float() * 2 / rope_half))
_row_angles = ROW_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)
_col_angles = COL_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)
_angles = torch.cat([_row_angles, _col_angles], dim=-1)
ROPE_COS = _angles.cos()
ROPE_SIN = _angles.sin()


def apply_rope(x, cos, sin):
    B, H, L, D = x.shape
    pairs = x.reshape(B, H, L, D // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack([out0, out1], dim=-1).reshape(B, H, L, D)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout_p = dropout

    def forward(self, x, rope_cos, rope_sin):
        h = self.norm1(x)
        B, L, D = h.shape
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout_p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.dropout1(self.out_proj(attn_out))
        h2 = self.norm2(x)
        h2 = self.linear2(self.dropout(F.relu(self.linear1(h2))))
        x = x + self.dropout2(h2)
        return x


class SudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, 9)
        # Q-head: predicts "is the puzzle solved?" from mean-pooled hidden state
        self.q_head = nn.Linear(d_model, 1)
        # Init: zero weight, large negative bias → default "not done"
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

    def forward(self, x, return_all=False, n_iters=None):
        if n_iters is None:
            n_iters = n_iterations
        batch_size = x.size(0)
        device = x.device
        rope_cos = ROPE_COS.to(device)
        rope_sin = ROPE_SIN.to(device)

        h_prev = self.initial_encoder(x)
        preds = torch.zeros(batch_size, 81, 9, device=device)

        all_logits = []
        all_q_logits = []
        for _ in range(n_iters):
            h = h_prev + self.pred_proj(preds)
            for layer in self.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)

            if return_all:
                all_logits.append(logits)
                q_logit = self.q_head(h.mean(dim=1))  # (B, 1)
                all_q_logits.append(q_logit.squeeze(-1))  # (B,)

        if return_all:
            return all_logits, all_q_logits
        return logits


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

    print(f"\nExperiment: Q-head + 2D RoPE (32 iters)")
    print(f"Architecture: d_model={d_model}, d_ff={d_ff}, n_layers={n_layers}")
    print(f"Q-head loss weight: {q_loss_weight}")
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
        all_logits, all_q_logits = model(x_batch, return_all=True)
        mask = x_batch[:, :, 0].to(dtype=torch.float32)
        t_batch = t_batch.to(dtype=torch.long)

        # Standard cross-entropy loss over all iterations
        ce_loss = 0
        for logits in all_logits:
            per_cell = F.cross_entropy(logits.reshape(-1, 9), t_batch.reshape(-1), reduction='none')
            per_cell = per_cell.view(t_batch.size(0), 81)
            ce_loss = ce_loss + (per_cell * mask).sum() / mask.sum()
        ce_loss = ce_loss / len(all_logits)

        # Q-head BCE loss: target = "are all empty cells correct?"
        q_loss = 0
        for it, (logits, q_logit) in enumerate(zip(all_logits, all_q_logits)):
            preds = logits.argmax(dim=-1)  # (B, 81)
            all_correct = ((preds == t_batch) | (mask == 0)).all(dim=1).float()  # (B,)
            q_loss = q_loss + F.binary_cross_entropy_with_logits(
                q_logit.float(), all_correct, reduction='mean')
        q_loss = q_loss / len(all_logits)

        total_loss = ce_loss + q_loss_weight * q_loss
        return total_loss, ce_loss, q_loss, all_logits, all_q_logits, mask, t_batch

    def eval_with_iters(n_iters, use_qhead_stop=False):
        """Evaluate with given iteration count. If use_qhead_stop, take predictions
        from the first iteration where Q-head says 'done' instead of the last."""
        model.eval()
        results = {}
        total_solved = 0
        total_puzzles = 0
        total_q_correct = 0
        total_q_count = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for name, data in test_data.items():
                x_test = data['x']
                puzzles = data['puzzles']
                solutions = data['solutions']
                puzzles_solved = 0
                for start in range(0, len(puzzles), 256):
                    end = min(start + 256, len(puzzles))
                    batch_x = x_test[start:end]
                    all_logits, all_q_logits = model(batch_x, return_all=True, n_iters=n_iters)
                    B = batch_x.size(0)

                    if use_qhead_stop:
                        # Per puzzle: take logits from first iter where q >= 0, else last
                        # Stack: (n_iters, B, 81, 9) and (n_iters, B)
                        stacked_logits = torch.stack(all_logits)  # (T, B, 81, 9)
                        stacked_q = torch.stack(all_q_logits)  # (T, B)
                        halt_mask = stacked_q >= 0  # (T, B)
                        # First halt iter per puzzle (T if never halts)
                        first_halt = torch.full((B,), n_iters - 1, device=batch_x.device, dtype=torch.long)
                        for t in range(n_iters):
                            newly_halted = halt_mask[t] & (first_halt == n_iters - 1)
                            first_halt[newly_halted] = t
                        # Gather logits at halt iteration for each puzzle
                        preds_full = stacked_logits[first_halt, torch.arange(B, device=batch_x.device)].argmax(dim=-1).cpu()
                    else:
                        preds_full = all_logits[-1].argmax(dim=-1).cpu()

                    q_pred = (all_q_logits[-1] >= 0).cpu()

                    for b, (puzzle, solution) in enumerate(zip(puzzles[start:end], solutions[start:end])):
                        pred_solution = list(puzzle)
                        for i in range(81):
                            if puzzle[i] == '.':
                                pred_solution[i] = str(preds_full[b, i].item() + 1)
                        is_solved = ''.join(pred_solution) == solution
                        if is_solved:
                            puzzles_solved += 1
                        if q_pred[b].item() == is_solved:
                            total_q_correct += 1
                        total_q_count += 1

                results[name] = {'solved': puzzles_solved, 'total': len(puzzles)}
                total_solved += puzzles_solved
                total_puzzles += len(puzzles)
        results['_total'] = {'solved': total_solved, 'total': total_puzzles}
        results['_q_acc'] = total_q_correct / total_q_count if total_q_count > 0 else 0
        return results

    def evaluate_all():
        return eval_with_iters(n_iterations)

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
            total_loss, ce_loss, q_loss, all_logits, all_q_logits, mask, t_batch = compute_loss(x_batch, t_batch)
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                final_logits = all_logits[-1]
                preds = final_logits.argmax(dim=-1)
                correct = (preds == t_batch) & (mask > 0)
                train_acc = correct.sum().item() / mask.sum().item()

                # Q-head train accuracy at final iteration
                all_correct = ((preds == t_batch) | (mask == 0)).all(dim=1)
                q_pred = all_q_logits[-1] >= 0
                q_acc = (q_pred == all_correct).float().mean().item()

            do_eval = step % eval_every == 0 or step == total_steps - 1
            if do_eval:
                results = evaluate_all()
                total_r = results.pop('_total')
                q_test_acc = results.pop('_q_acc')
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {ce_loss.item():.4f}+{q_loss.item():.4f} Acc: {train_acc:.2%} Q:{q_acc:.2%}/{q_test_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {ce_loss.item():.4f}+{q_loss.item():.4f} Acc: {train_acc:.2%} Q:{q_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - exp_qhead_32")
    log("="*60)

    # Standard eval at 32 iters
    log("\n--- 32 iters (training default) ---")
    results = eval_with_iters(32)
    total_r = results.pop('_total')
    q_test_acc = results.pop('_q_acc')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    log(f"Q-head test accuracy: {q_test_acc:.2%}")

    # 48 iters, always take last
    log("\n--- 48 iters (no Q-head stop) ---")
    results = eval_with_iters(48)
    total_r = results.pop('_total')
    results.pop('_q_acc')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")

    # 48 iters, Q-head early stop
    log("\n--- 48 iters (Q-head early stop) ---")
    results = eval_with_iters(48, use_qhead_stop=True)
    total_r = results.pop('_total')
    results.pop('_q_acc')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")

    log(f"\n2D RoPE baseline (16 iters): 82.5%")
    log(f"2D RoPE baseline (32 test iters): 88.4%")

    final_path = os.path.join(output_dir, "model_qhead_32.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()


if __name__ == "__main__":
    train()
