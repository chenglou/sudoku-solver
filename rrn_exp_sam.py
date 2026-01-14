# RRN with SAM (Sharpness-Aware Minimization)
# Hypothesis: SAM finds flatter minima, improving generalization
# Based on "Sharpness-Aware Minimization for Efficiently Improving Generalization"

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

torch.set_float32_matmul_precision('high')

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper.

    1. Compute gradient g at weights w
    2. Perturb: w' = w + rho * g / ||g||
    3. Compute gradient g' at perturbed w'
    4. Update w using g' (not g)
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
                self.state[p]['old_w'] = p.data.clone()
                p.data.add_(p.grad, alpha=scale)

    @torch.no_grad()
    def second_step(self):
        """Restore weights and apply update using perturbed gradient"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]['old_w'])
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
hidden_size = 128
num_steps = 16
lr = 1e-3
steps = 100000
n_train = 100000
n_test = 1000
batch_size = 64  # Keep same batch size first, test SAM alone

def build_sudoku_edges():
    edges = []
    for i in range(81):
        row_i, col_i = i // 9, i % 9
        box_i = (row_i // 3) * 3 + (col_i // 3)
        for j in range(81):
            if i == j:
                continue
            row_j, col_j = j // 9, j % 9
            box_j = (row_j // 3) * 3 + (col_j // 3)
            if row_i == row_j or col_i == col_j or box_i == box_j:
                edges.append((j, i))
    return torch.tensor(edges, dtype=torch.long).t()

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class SudokuRRN(nn.Module):
    def __init__(self, edge_index):
        super().__init__()
        self.register_buffer('edge_index', edge_index)
        self.input_proj = nn.Linear(10, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(81, hidden_size) * 0.02)
        self.message_mlp = MLP(hidden_size * 2, hidden_size, hidden_size)
        self.node_mlp = MLP(hidden_size * 3, hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, 9)

    def forward(self, x, return_all=False):
        batch_size, num_nodes, _ = x.shape
        device = x.device
        x_embed = self.input_proj(x) + self.pos_embed
        h = x_embed
        src, dst = self.edge_index
        all_logits = []
        for _ in range(num_steps):
            h_src = h[:, src]
            h_dst = h[:, dst]
            messages = self.message_mlp(torch.cat([h_src, h_dst], dim=-1))
            aggregated = torch.zeros(batch_size, num_nodes, hidden_size, device=device, dtype=messages.dtype)
            dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_size)
            aggregated.scatter_add_(1, dst_expanded, messages)
            h = h + self.node_mlp(torch.cat([h, x_embed, aggregated], dim=-1))
            h = self.layer_norm(h)
            if return_all:
                all_logits.append(self.output_head(h))
        if return_all:
            return all_logits
        return self.output_head(h)

def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1
        else:
            x[i, 1 + int(c) - 1] = 1
    return x

def get_targets(puzzle_str, solution_str):
    holes, targets = [], []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.':
            holes.append(i)
            targets.append(int(s) - 1)
    return torch.tensor(holes), torch.tensor(targets)

edge_index = build_sudoku_edges()
print(f"Sudoku graph: 81 nodes, {edge_index.shape[1]} edges")

df = pd.read_csv("data/sudoku-3m.csv")
easy_df = df[df['difficulty'] == df['difficulty'].min()]
print(f"Puzzles at easiest difficulty: {len(easy_df)}")
df = easy_df.head(n_train + n_test)

train_puzzles = df['puzzle'].tolist()[:n_train]
train_solutions = df['solution'].tolist()[:n_train]
test_puzzles = df['puzzle'].tolist()[n_train:]
test_solutions = df['solution'].tolist()[n_train:]
print(f"Train: {len(train_puzzles)}, Test: {len(test_puzzles)}")

x_train = torch.stack([encode_puzzle(p) for p in train_puzzles])
train_holes = [get_targets(p, s) for p, s in zip(train_puzzles, train_solutions)]
x_test = torch.stack([encode_puzzle(p) for p in test_puzzles])
test_holes = [get_targets(p, s) for p, s in zip(test_puzzles, test_solutions)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuRRN(edge_index).to(device)
x_train = x_train.to(device)
x_test = x_test.to(device)

train_holes = [(h[0].to(device), h[1].to(device)) for h in train_holes]
train_holes_count = [len(h[0]) for h in train_holes]
test_holes = [(h[0].to(device), h[1].to(device)) for h in test_holes]
test_holes_count = [len(h[0]) for h in test_holes]

# SAM wrapping AdamW (no GradScaler - bf16 doesn't need it and it conflicts with SAM)
optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=lr, betas=(0.9, 0.95))

def evaluate_test():
    model.eval()
    total_loss, total_correct, total_cells, puzzles_solved = 0, 0, 0, 0
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
            total_correct += (logits_holes.argmax(dim=-1) == targets).sum().item()
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

print(f"\nRRN + SAM (rho=0.05)")
print(f"Training on {device}...")

log_file = open("rrn_exp_sam.log", "w")
log_file.write("step,train_loss,train_acc,test_loss,test_acc,solved\n")

for step in range(steps):
    model.train()
    optimizer.zero_grad()

    batch_idx = torch.randperm(n_train)[:batch_size]
    x_batch = x_train[batch_idx]
    batch_list = batch_idx.tolist()
    hole_c = torch.cat([train_holes[i][0] for i in batch_list])
    targets = torch.cat([train_holes[i][1] for i in batch_list])
    counts = torch.tensor([train_holes_count[i] for i in batch_list], device=device)
    hole_b = torch.repeat_interleave(torch.arange(len(batch_list), device=device), counts)

    # SAM first step: compute gradient at current weights
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        all_logits = model(x_batch, return_all=True)
        loss = sum(F.cross_entropy(logits[hole_b, hole_c], targets) for logits in all_logits) / len(all_logits)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.first_step()  # perturb weights

    # SAM second step: compute gradient at perturbed weights
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        all_logits = model(x_batch, return_all=True)
        loss = sum(F.cross_entropy(logits[hole_b, hole_c], targets) for logits in all_logits) / len(all_logits)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.second_step()  # restore weights and update

    if step % 100 == 0 or step == steps - 1:
        with torch.no_grad():
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()
        train_loss = loss.item()

        if step % 1000 == 0 or step == steps - 1:
            test_loss, test_acc, puzzles_solved = evaluate_test()
            print(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Test Loss: {test_loss:.4f} Acc: {test_acc:.2%} | Solved: {puzzles_solved}/{n_test}")
            log_file.write(f"{step},{train_loss:.6f},{train_acc:.6f},{test_loss:.6f},{test_acc:.6f},{puzzles_solved}\n")
        else:
            print(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%}")
            log_file.write(f"{step},{train_loss:.6f},{train_acc:.6f},,,\n")
        log_file.flush()

log_file.close()
test_loss, test_acc, puzzles_solved = evaluate_test()
print(f"\nFinal Test: Loss {test_loss:.4f} | Acc {test_acc:.1%} | {puzzles_solved}/{n_test} puzzles solved")
