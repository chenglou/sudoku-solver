# Experiment: Unrolled - 16 separate 4-layer transformers (one per iteration)
# Tests pure weight sharing: same FLOPs (64 layer passes), 16x more params
# If unrolled wins: weight sharing hurts
# If same: weight sharing saves 16x memory for free
# If unrolled loses: weight sharing acts as regularization
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if g.ndim >= 2:
                    g = self._newton_schulz(g, ns_steps)
                    g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr)

    @staticmethod
    def _newton_schulz(G, steps=5):
        shape = G.shape
        if G.ndim > 2:
            G = G.view(G.size(0), -1)
        G = G / (G.norm() + 1e-7)
        for _ in range(steps):
            A = G @ G.T
            G = 1.5 * G - 0.5 * A @ G
        return G.view(shape)

# Hyperparameters
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4  # Same as baseline
n_iterations = 16  # 16 separate transformers, one per iteration
lr = 1e-3
steps = 100000
n_train = 100000
n_test = 1000
batch_size = 128

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

        # 16 separate transformers (4 layers each) - one per iteration
        self.transformers = nn.ModuleList()
        for _ in range(n_iterations):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                batch_first=True, norm_first=True,
            )
            self.transformers.append(
                nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            )

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

        # Each iteration uses a different transformer
        for i, transformer in enumerate(self.transformers):
            x_in = torch.cat([x, preds], dim=-1)
            h = self.input_proj(x_in)
            h = h + pos_embed
            h = transformer(h)
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
            x[i, 1 + int(c) - 1] = 1
    return x

def get_targets(puzzle_str, solution_str):
    holes, targets = [], []
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

print(f"Train: {len(train_puzzles)}, Test: {len(test_puzzles)}")

x_train = torch.stack([encode_puzzle(p) for p in train_puzzles])
train_holes = [get_targets(p, s) for p, s in zip(train_puzzles, train_solutions)]
x_test = torch.stack([encode_puzzle(p) for p in test_puzzles])
test_holes = [get_targets(p, s) for p, s in zip(test_puzzles, test_solutions)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformer().to(device)

# Print param count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} (vs ~800k for shared)")

model = torch.compile(model)
x_train = x_train.to(device)
x_test = x_test.to(device)

train_holes = [(h[0].to(device), h[1].to(device)) for h in train_holes]
train_holes_count = [len(h[0]) for h in train_holes]
test_holes = [(h[0].to(device), h[1].to(device)) for h in test_holes]
test_holes_count = [len(h[0]) for h in test_holes]

muon_params, adamw_params = [], []
for name, param in model.named_parameters():
    if param.ndim >= 2:
        muon_params.append(param)
    else:
        adamw_params.append(param)

optimizer_muon = Muon(muon_params, lr=0.02, momentum=0.95)
optimizer_adamw = torch.optim.AdamW(adamw_params, lr=lr, betas=(0.9, 0.95))

def evaluate_test():
    model.eval()
    total_loss, total_correct, total_cells, puzzles_solved = 0, 0, 0, 0
    with torch.no_grad():
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

print(f"\nEXPERIMENT: Unrolled (16 separate 4-layer transformers)")
print(f"Training on {device}...")

log_file = open("exp_unrolled.log", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

for step in range(steps):
    model.train()
    optimizer_muon.zero_grad()
    optimizer_adamw.zero_grad()

    batch_idx = torch.randperm(n_train)[:batch_size]
    x_batch = x_train[batch_idx]

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

    loss.backward()
    optimizer_muon.step()
    optimizer_adamw.step()

    if step % 100 == 0 or step == steps - 1:
        with torch.no_grad():
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()
        train_loss = loss.item()

        if step % 1000 == 0 or step == steps - 1:
            test_loss, test_acc, puzzles_solved = evaluate_test()
            log(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Test Loss: {test_loss:.4f} Acc: {test_acc:.2%} | Solved: {puzzles_solved}/{n_test}")
        else:
            log(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%}")

test_loss, test_acc, puzzles_solved = evaluate_test()
log(f"\nFinal Test: Loss {test_loss:.4f} | Acc {test_acc:.1%} | {puzzles_solved}/{n_test} puzzles solved")
log_file.close()
