# Zero-shot evaluation: Train on easy (difficulty 0), test on all difficulties
# Uses our best config: SAM + BS=512

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

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


# Hyperparameters (best config)
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16
lr = 1e-3
steps = 100000
n_train = 100000
batch_size = 512
sam_rho = 0.05

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


def evaluate_on_difficulty(model, df, difficulty_min, difficulty_max, n_test=1000, device='cuda'):
    """Evaluate model on puzzles within difficulty range"""
    subset = df[(df['difficulty'] >= difficulty_min) & (df['difficulty'] < difficulty_max)]
    if len(subset) < n_test:
        n_test = len(subset)
        print(f"  (Only {n_test} puzzles available)")

    subset = subset.head(n_test)
    puzzles = subset['puzzle'].tolist()
    solutions = subset['solution'].tolist()

    x_test = torch.stack([encode_puzzle(p) for p in puzzles]).to(device)

    model.eval()
    puzzles_solved = 0
    total_correct = 0
    total_cells = 0

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            x_batch = x_test[start:end]
            logits = model(x_batch)
            preds_full = logits.argmax(dim=-1).cpu()

            for b, (puzzle, solution) in enumerate(zip(puzzles[start:end], solutions[start:end])):
                pred_solution = list(puzzle)
                correct = 0
                holes = 0
                for i in range(81):
                    if puzzle[i] == '.':
                        pred_solution[i] = str(preds_full[b, i].item() + 1)
                        holes += 1
                        if pred_solution[i] == solution[i]:
                            correct += 1
                total_correct += correct
                total_cells += holes
                if ''.join(pred_solution) == solution:
                    puzzles_solved += 1

    acc = total_correct / total_cells if total_cells > 0 else 0
    return puzzles_solved, n_test, acc


# Load data
print("Loading data...")
df = pd.read_csv("data/sudoku-3m.csv")

# Training data: easiest difficulty
train_df = df[df['difficulty'] == df['difficulty'].min()].head(n_train)
print(f"Training on {len(train_df)} puzzles at difficulty {train_df['difficulty'].iloc[0]}")

train_puzzles = train_df['puzzle'].tolist()
train_solutions = train_df['solution'].tolist()

x_train = torch.stack([encode_puzzle(p) for p in train_puzzles])
train_holes = [get_targets(p, s) for p, s in zip(train_puzzles, train_solutions)]
train_holes_count = [len(h[0]) for h in train_holes]

device = torch.device("cuda")
model = SudokuTransformer().to(device)
model = torch.compile(model)
x_train = x_train.to(device)
train_holes = [(h[0].to(device), h[1].to(device)) for h in train_holes]

optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=sam_rho, lr=lr, betas=(0.9, 0.95))

print(f"\nTraining with SAM + BS={batch_size}...")

log_file = open("train_log_eval_diff.txt", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def compute_loss(x_batch, batch_idx):
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

    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss1, _, _, _, _ = compute_loss(x_batch, batch_idx)
    loss1.backward()
    optimizer.first_step()

    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss2, all_logits, hole_b, hole_c, targets = compute_loss(x_batch, batch_idx)
    loss2.backward()
    optimizer.second_step()

    if step % 1000 == 0 or step == steps - 1:
        with torch.no_grad():
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()
        log(f"Step {step:5d} | Train Loss: {loss2.item():.4f} Acc: {train_acc:.2%}")

# Final evaluation on all difficulties
log("\n" + "="*60)
log("ZERO-SHOT EVALUATION ON DIFFERENT DIFFICULTIES")
log("(Model trained on difficulty 0.0 only)")
log("="*60)

difficulties = [
    (0.0, 0.1, "0.0 (train)"),
    (1.0, 2.0, "1.x"),
    (2.0, 3.0, "2.x"),
    (3.0, 4.0, "3.x"),
    (4.0, 5.0, "4.x"),
    (5.0, 10.0, "5.x+"),
]

for d_min, d_max, label in difficulties:
    solved, total, acc = evaluate_on_difficulty(model, df, d_min, d_max, n_test=1000, device=device)
    log(f"Difficulty {label:12s}: {solved:4d}/{total} solved ({solved/total*100:5.1f}%), cell acc: {acc:.1%}")

# Save model checkpoint
torch.save(model.state_dict(), "model_sam_bs512.pt")
print("Model saved to model_sam_bs512.pt")

log_file.close()
print("\nDone! Results saved to train_log_eval_diff.txt")
