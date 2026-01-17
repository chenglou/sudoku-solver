# Reverse Curriculum learning: Hard → Easy
# Train in phases, gradually decreasing difficulty (anti-curriculum)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random

torch.set_float32_matmul_precision('high')

# SAM optimizer
class SAM(torch.optim.Optimizer):
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


# Hyperparameters
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16
lr = 1e-3
total_steps = 100000
batch_size = 512
sam_rho = 0.05

# Reverse curriculum phases - 20k steps each, progressively easier
# (start_step, end_step, min_difficulty, name)
PHASES = [
    (0, 20000, 3.0, "Phase 1: Hard only (3.0+)"),
    (20000, 40000, 2.0, "Phase 2: Medium-Hard+ (2.0+)"),
    (40000, 60000, 1.0, "Phase 3: Medium+ (1.0+)"),
    (60000, 80000, 0.0, "Phase 4: All (0.0+)"),
    (80000, 100000, 0.0, "Phase 5: All (0.0+)"),
]

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

    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        device = x.device
        row_idx = ROW_IDX.to(device)
        col_idx = COL_IDX.to(device)
        box_idx = BOX_IDX.to(device)
        pos_embed = self.row_embed(row_idx) + self.col_embed(col_idx) + self.box_embed(box_idx)
        preds = torch.zeros(batch_size, 81, 9, device=device)
        all_logits = []
        for _ in range(n_iterations):
            x_in = torch.cat([x, preds], dim=-1)
            h = self.input_proj(x_in)
            h = h + pos_embed
            h = self.transformer(h)
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
            if return_all:
                all_logits.append(logits)
        return all_logits if return_all else logits


def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x


def get_targets(puzzle_str, solution_str):
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.':
            holes.append(i)
            targets.append(int(s) - 1)
    return torch.tensor(holes), torch.tensor(targets)


device = torch.device("cuda")

# Load all data
print("Loading data...")
df = pd.read_csv("data/sudoku-3m.csv")
print(f"Total puzzles: {len(df)}")

# Pre-encode ALL training data on CPU (no limit!)
BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 10.0)]
bucket_data = {}

for min_diff, max_diff in BUCKETS:
    bucket_df = df[(df['difficulty'] >= min_diff) & (df['difficulty'] < max_diff)]
    # Exclude last 500 (reserved for test)
    train_pool = bucket_df.iloc[:-500]

    puzzles = train_pool['puzzle'].tolist()
    solutions = train_pool['solution'].tolist()

    print(f"  Encoding {min_diff}-{max_diff}: {len(puzzles)} puzzles...", end=" ", flush=True)

    # Pre-encode on CPU
    x_data = torch.stack([encode_puzzle(p) for p in puzzles])  # CPU
    holes_list = [get_targets(p, s) for p, s in zip(puzzles, solutions)]  # CPU
    holes_count = [len(h[0]) for h in holes_list]

    bucket_data[(min_diff, max_diff)] = {
        'x': x_data,  # CPU tensor
        'holes': holes_list,  # CPU tensors
        'holes_count': holes_count,
        'size': len(puzzles),
    }
    print("done")

# Build phase bucket lists
# For reverse curriculum: include buckets with d_min >= min_diff (harder puzzles first)
phase_buckets = {}
for start, end, min_diff, name in PHASES:
    buckets_for_phase = [k for k in bucket_data.keys() if k[0] >= min_diff]
    total = sum(bucket_data[k]['size'] for k in buckets_for_phase)
    phase_buckets[min_diff] = buckets_for_phase
    print(f"  {name}: {total} total puzzles from {len(buckets_for_phase)} buckets")

# Test set: 500 from each difficulty bucket (on GPU for fast eval)
print("\nPreparing test data...")
test_data = {}
for min_diff, max_diff in BUCKETS:
    name = f"{min_diff:.1f}" if max_diff < 5 else "4.x+"
    if name == "0.0":
        name = "0.0"
    elif name == "1.0":
        name = "1.x"
    elif name == "2.0":
        name = "2.x"
    elif name == "3.0":
        name = "3.x"
    else:
        name = "4.x+"

    bucket_df = df[(df['difficulty'] >= min_diff) & (df['difficulty'] < max_diff)]
    test_df = bucket_df.tail(500)

    puzzles = test_df['puzzle'].tolist()
    solutions = test_df['solution'].tolist()

    x_test = torch.stack([encode_puzzle(p) for p in puzzles]).to(device)

    test_data[name] = {
        'x': x_test,
        'puzzles': puzzles,
        'solutions': solutions,
    }
    print(f"  Test {name}: {len(puzzles)} puzzles")

model = SudokuTransformer().to(device)
model = torch.compile(model)

optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=sam_rho, lr=lr, betas=(0.9, 0.95))

print(f"\nReverse Curriculum Learning: Hard -> Easy")
print(f"Total steps: {total_steps}, BS={batch_size}, SAM rho={sam_rho}")

log_file = open("train_curriculum_reverse.txt", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def get_phase(step):
    """Get phase buckets for current step."""
    for start, end, min_diff, name in PHASES:
        if start <= step < end:
            return phase_buckets[min_diff], name
    return None, None


def sample_batch(active_buckets, batch_size):
    """Sample batch from active buckets, move to GPU."""
    # Calculate total size and weights
    sizes = [bucket_data[b]['size'] for b in active_buckets]
    total = sum(sizes)

    # Sample indices: which bucket and which index within bucket
    x_list = []
    holes_list = []
    counts_list = []

    for _ in range(batch_size):
        # Pick bucket proportionally
        r = random.randint(0, total - 1)
        cumsum = 0
        for b, s in zip(active_buckets, sizes):
            cumsum += s
            if r < cumsum:
                # Sample from this bucket
                idx = random.randint(0, s - 1)
                x_list.append(bucket_data[b]['x'][idx])
                holes_list.append(bucket_data[b]['holes'][idx])
                counts_list.append(bucket_data[b]['holes_count'][idx])
                break

    # Stack and move to GPU
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


def evaluate_all():
    """Evaluate on all difficulty levels."""
    model.eval()
    results = {}

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, data in test_data.items():
            x_test = data['x']
            puzzles = data['puzzles']
            solutions = data['solutions']

            logits = model(x_test)
            preds_full = logits.argmax(dim=-1).cpu()

            total_correct = 0
            total_cells = 0
            puzzles_solved = 0

            for b, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
                pred_solution = list(puzzle)
                n_correct = 0
                n_holes = 0
                for i in range(81):
                    if puzzle[i] == '.':
                        pred_digit = str(preds_full[b, i].item() + 1)
                        pred_solution[i] = pred_digit
                        n_holes += 1
                        if pred_digit == solution[i]:
                            n_correct += 1
                total_correct += n_correct
                total_cells += n_holes
                if ''.join(pred_solution) == solution:
                    puzzles_solved += 1

            results[name] = {
                'solved': puzzles_solved,
                'total': len(puzzles),
                'cell_acc': total_correct / total_cells if total_cells > 0 else 0,
            }

    return results


current_phase_name = None
current_buckets = None

for step in range(total_steps):
    buckets, phase_name = get_phase(step)

    if phase_name != current_phase_name:
        current_phase_name = phase_name
        current_buckets = buckets
        total_puzzles = sum(bucket_data[b]['size'] for b in buckets)
        log(f"\n{'='*60}")
        log(f"Step {step}: Entering {phase_name}")
        log(f"Training pool: {total_puzzles} puzzles")
        log(f"{'='*60}\n")

    model.train()

    # Sample batch from active buckets (CPU -> GPU)
    x_batch, holes_batch, counts_batch = sample_batch(current_buckets, batch_size)

    # SAM step 1
    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss1, _, _, _, _ = compute_loss(x_batch, holes_batch, counts_batch)
    loss1.backward()
    optimizer.first_step()

    # SAM step 2
    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss2, all_logits, hole_b, hole_c, targets = compute_loss(x_batch, holes_batch, counts_batch)
    loss2.backward()
    optimizer.second_step()

    # Logging
    if step % 100 == 0 or step == total_steps - 1:
        with torch.no_grad():
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()

        if step % 5000 == 0 or step == total_steps - 1:
            results = evaluate_all()
            total_solved = sum(r['solved'] for r in results.values())
            log(f"Step {step:5d} | Loss: {loss2.item():.4f} Acc: {train_acc:.2%} | " +
                " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                f" | Total: {total_solved}/2500")
        else:
            log(f"Step {step:5d} | Loss: {loss2.item():.4f} Acc: {train_acc:.2%}")

# Final evaluation
log("\n" + "="*60)
log("FINAL RESULTS - Reverse Curriculum Learning (Hard -> Easy)")
log("="*60)
results = evaluate_all()
for name, r in results.items():
    log(f"Difficulty {name:6s}: {r['solved']:3d}/{r['total']} solved ({100*r['solved']/r['total']:5.1f}%), cell acc: {100*r['cell_acc']:.1f}%")

total_solved = sum(r['solved'] for r in results.values())
log(f"\nTotal: {total_solved}/2500 solved")

# Save model
torch.save({k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}, "model_curriculum_reverse.pt")
log("Model saved to model_curriculum_reverse.pt")

log_file.close()
