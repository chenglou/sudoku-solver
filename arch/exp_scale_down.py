# Experiment: Scale DOWN the model
# Half the dimensions to test if smaller model works

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import multiprocessing as mp
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


# Hyperparameters - SCALED DOWN
d_model = 64       # was 128
n_heads = 2        # was 4
d_ff = 256         # was 512
n_layers = 2       # was 4
n_iterations = 16
lr = 1e-3
total_steps = 70000
batch_size = 512
sam_rho = 0.05
train_size = 2700000

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


# Fast data loading functions
def encode_puzzle_np(puzzle_str):
    x = np.zeros((81, 10), dtype=np.float32)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x

def encode_and_targets(args):
    puzzle_str, solution_str = args
    x = encode_puzzle_np(puzzle_str)
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.':
            holes.append(i)
            targets.append(int(s) - 1)
    return x, np.array(holes, dtype=np.int64), np.array(targets, dtype=np.int64)


device = torch.device("cuda")

print("Loading sudoku-extreme train split...")
dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
print(f"Total available: {len(dataset)}")
print(f"Using first {train_size} for training")

test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
print(f"Test set: {len(test_dataset)}")

# Fast batch extraction
print("\nEncoding training data (fast parallel loading)...")
import time
t0 = time.time()
batch = dataset[:train_size]
ratings = np.array(batch['rating'])
questions = batch['question']
answers = batch['answer']
print(f"  Batch extraction: {time.time()-t0:.1f}s")

train_data = {}
n_workers = mp.cpu_count()
print(f"  Using {n_workers} workers for encoding...")

for min_r, max_r, name in RATING_BUCKETS:
    t0 = time.time()
    mask = (ratings >= min_r) & (ratings <= max_r)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        continue

    puzzles = [questions[i] for i in indices]
    solutions = [answers[i] for i in indices]

    with mp.Pool(n_workers) as pool:
        results = pool.map(encode_and_targets, zip(puzzles, solutions))

    x_list = [torch.from_numpy(r[0]) for r in results]
    holes_list = [(torch.from_numpy(r[1]), torch.from_numpy(r[2])) for r in results]
    holes_count = [len(r[1]) for r in results]
    x_data = torch.stack(x_list)

    train_data[(min_r, max_r)] = {
        'x': x_data,
        'holes': holes_list,
        'holes_count': holes_count,
        'size': len(puzzles),
    }
    print(f"  Rating {name}: {len(puzzles)} puzzles ({time.time()-t0:.1f}s)")

# Build phase buckets
phase_buckets = {}
for start, end, min_rating, name in PHASES:
    buckets_for_phase = [k for k in train_data.keys() if k[0] >= min_rating]
    total = sum(train_data[k]['size'] for k in buckets_for_phase)
    phase_buckets[min_rating] = buckets_for_phase
    print(f"  {name}: {total} puzzles")

# Prepare test data
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

    with mp.Pool(n_workers) as pool:
        results = pool.map(encode_and_targets, zip(puzzles, solutions))
    x_test = torch.stack([torch.from_numpy(r[0]) for r in results]).to(device)

    test_data[name] = {
        'x': x_test,
        'puzzles': puzzles,
        'solutions': solutions,
    }
    print(f"  Test {name}: {len(puzzles)} puzzles")

model = SudokuTransformer().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,} (baseline: ~800K)")

model = torch.compile(model)

optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=sam_rho, lr=lr, betas=(0.9, 0.95))

print(f"\nExperiment: Scale DOWN")
print(f"Architecture: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}")
print(f"Iterations: {n_iterations}, BS={batch_size}, SAM rho={sam_rho}")

log_file = open("exp_scale_down.log", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def get_phase(step):
    for start, end, min_rating, name in PHASES:
        if start <= step < end:
            return phase_buckets[min_rating], name
    return None, None


def sample_batch(active_buckets, batch_size):
    sizes = [train_data[b]['size'] for b in active_buckets]
    total = sum(sizes)
    x_list = []
    holes_list = []
    counts_list = []
    for _ in range(batch_size):
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

            results[name] = {
                'solved': puzzles_solved,
                'total': len(puzzles),
            }
            total_solved += puzzles_solved
            total_puzzles += len(puzzles)
    results['_total'] = {'solved': total_solved, 'total': total_puzzles}
    return results


current_phase_name = None
current_buckets = None

for step in range(total_steps):
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
        loss2, all_logits, hole_b, hole_c, targets = compute_loss(x_batch, holes_batch, counts_batch)
    loss2.backward()
    optimizer.second_step()

    if step % 100 == 0 or step == total_steps - 1:
        with torch.no_grad():
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()

        if step % 5000 == 0 or step == total_steps - 1:
            results = evaluate_all()
            total_r = results.pop('_total')
            log(f"Step {step:5d} | Loss: {loss2.item():.4f} Acc: {train_acc:.2%} | " +
                " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
        else:
            log(f"Step {step:5d} | Loss: {loss2.item():.4f} Acc: {train_acc:.2%}")

log("\n" + "="*60)
log("FINAL RESULTS - Scale DOWN")
log("="*60)
results = evaluate_all()
total_r = results.pop('_total')
for name, r in results.items():
    log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
log(f"Baseline (d=128, L=4): 71.4%")

torch.save({k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}, "model_scale_down.pt")
log("Model saved to model_scale_down.pt")

log_file.close()
