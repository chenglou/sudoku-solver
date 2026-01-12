# SAM (Sharpness-Aware Minimization) experiment
# Hypothesis: SAM finds flatter minima, closing the generalization gap for large batches
# Testing with BS=512 to see if it can match BS=256's results

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

torch.set_float32_matmul_precision('high')

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper.

    1. Compute gradient g at weights w
    2. Compute epsilon = rho * g / ||g||
    3. Compute gradient g' at w + epsilon (the "sharpness" direction)
    4. Update w using g'
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho)
        params = list(params)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        """Perturb weights: w -> w + epsilon"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            rho = group.get('rho', 0.05)
            scale = rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                # Store original weights
                self.state[p]['old_w'] = p.data.clone()
                # Perturb: w + rho * grad / ||grad||
                p.data.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        """Restore weights and apply update using perturbed gradient"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Restore original weights
                p.data.copy_(self.state[p]['old_w'])
        # Now base optimizer updates using gradient computed at perturbed point
        self.base_optimizer.step()

    @torch.no_grad()
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm()
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ])
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
steps = 100000
n_train = 100000
n_test = 1000
batch_size = 512  # Large batch to test SAM's effect on generalization gap
sam_rho = 0.05    # SAM perturbation radius

# Precompute row/col/box indices
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
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=True,
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

        if return_all:
            return all_logits
        return logits

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

df = pd.read_csv("data/sudoku-3m.csv")
easy_df = df[df['difficulty'] == df['difficulty'].min()]
print(f"Puzzles at easiest difficulty: {len(easy_df)} (need {n_train + n_test})")
df = easy_df.head(n_train + n_test)

train_puzzles = df['puzzle'].tolist()[:n_train]
train_solutions = df['solution'].tolist()[:n_train]
test_puzzles = df['puzzle'].tolist()[n_train:]
test_solutions = df['solution'].tolist()[n_train:]

print(f"Train: {len(train_puzzles)}, Test: {len(test_puzzles)} (difficulty={df['difficulty'].iloc[0]})")

x_train = torch.stack([encode_puzzle(p) for p in train_puzzles])
train_holes = [get_targets(p, s) for p, s in zip(train_puzzles, train_solutions)]

x_test = torch.stack([encode_puzzle(p) for p in test_puzzles])
test_holes = [get_targets(p, s) for p, s in zip(test_puzzles, test_solutions)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformer().to(device)
model = torch.compile(model)
x_train = x_train.to(device)
x_test = x_test.to(device)

train_holes = [(h[0].to(device), h[1].to(device)) for h in train_holes]
train_holes_count = [len(h[0]) for h in train_holes]
test_holes = [(h[0].to(device), h[1].to(device)) for h in test_holes]
test_holes_count = [len(h[0]) for h in test_holes]

# Use SAM with AdamW as base optimizer
optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=sam_rho, lr=lr, betas=(0.9, 0.95))

def evaluate_test():
    model.eval()
    total_loss = 0
    total_correct = 0
    total_cells = 0
    puzzles_solved = 0

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_batch = x_test[start:end]
            logits = model(x_batch)

            batch_range = list(range(start, end))
            hole_c = torch.cat([test_holes[i][0] for i in batch_range])
            targets = torch.cat([test_holes[i][1] for i in batch_range])
            counts = torch.tensor([test_holes_count[i] for i in batch_range], device=device)
            hole_b = torch.repeat_interleave(torch.arange(len(batch_range), device=device), counts)

            logits_holes = logits[hole_b, hole_c]
            total_loss += F.cross_entropy(logits_holes, targets, reduction='sum').item()
            preds = logits_holes.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_cells += len(targets)

            preds_full = logits.argmax(dim=-1).cpu()
            for b, (puzzle, solution) in enumerate(zip(test_puzzles[start:end], test_solutions[start:end])):
                pred_solution = list(puzzle)
                for i in range(81):
                    if puzzle[i] == '.':
                        pred_solution[i] = str(preds_full[b, i].item() + 1)
                if ''.join(pred_solution) == solution:
                    puzzles_solved += 1

    return total_loss / total_cells, total_correct / total_cells, puzzles_solved

total_holes = sum(len(h[0]) for h in train_holes)
print(f"Total empty cells in train: {total_holes}")
print(f"\nTraining with SAM (rho={sam_rho}) on {device}...")
print(f"Batch size: {batch_size}, expecting ~2x slower than vanilla due to SAM")

log_file = open("train_log_sam.txt", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

def compute_loss(x_batch, batch_idx):
    """Compute loss for a batch - used twice per SAM step"""
    all_logits = model(x_batch, return_all=True)

    batch_list = batch_idx.tolist()
    hole_c = torch.cat([train_holes[i][0] for i in batch_list])
    targets = torch.cat([train_holes[i][1] for i in batch_list])
    counts = torch.tensor([train_holes_count[i] for i in batch_list], device=device)
    hole_b = torch.repeat_interleave(torch.arange(len(batch_list), device=device), counts)

    loss = 0
    for logits in all_logits:
        logits_holes = logits[hole_b, hole_c]
        loss = loss + F.cross_entropy(logits_holes, targets)
    loss = loss / len(all_logits)

    return loss, all_logits, hole_b, hole_c, targets

for step in range(steps):
    model.train()

    batch_idx = torch.randperm(n_train)[:batch_size]
    x_batch = x_train[batch_idx]

    # SAM Step 1: Compute gradient and perturb
    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss1, _, _, _, _ = compute_loss(x_batch, batch_idx)
    loss1.backward()
    optimizer.first_step()  # perturb weights

    # SAM Step 2: Compute gradient at perturbed point and update
    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss2, all_logits, hole_b, hole_c, targets = compute_loss(x_batch, batch_idx)
    loss2.backward()
    optimizer.second_step()  # restore and update

    if step % 100 == 0 or step == steps - 1:
        with torch.no_grad():
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()
        train_loss = loss2.item()

        if step % 1000 == 0 or step == steps - 1:
            test_loss, test_acc, puzzles_solved = evaluate_test()
            log(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Test Loss: {test_loss:.4f} Acc: {test_acc:.2%} | Solved: {puzzles_solved}/{n_test}")
        else:
            log(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%}")

test_loss, test_acc, puzzles_solved = evaluate_test()
log(f"\nFinal Test: Loss {test_loss:.4f} | Acc {test_acc:.1%} | {puzzles_solved}/{n_test} puzzles solved")
log_file.close()
