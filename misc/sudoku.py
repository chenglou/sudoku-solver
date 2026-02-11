# Iterative transformer for solving Sudoku puzzles
# Key components: iterative refinement (16 steps), intermediate supervision, structured pos encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Speed optimizations
torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere+ GPUs

class Muon(torch.optim.Optimizer):
    """Muon optimizer - Momentum Orthogonalized by Newton-Schulz."""
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

                # Newton-Schulz orthogonalization for 2D+ params
                if g.ndim >= 2:
                    g = self._newton_schulz(g, ns_steps)
                    # Scale by sqrt of matrix dimensions
                    g = g * max(1, g.size(0) / g.size(1)) ** 0.5

                p.data.add_(g, alpha=-lr)

    @staticmethod
    def _newton_schulz(G, steps=5):
        """Approximate orthogonalization via Newton-Schulz iteration."""
        # Reshape to 2D for orthogonalization
        shape = G.shape
        if G.ndim > 2:
            G = G.view(G.size(0), -1)

        # Normalize
        G = G / (G.norm() + 1e-7)

        # Newton-Schulz iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
        for _ in range(steps):
            A = G @ G.T
            G = 1.5 * G - 0.5 * A @ G

        return G.view(shape)

# Hyperparameters
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16  # Number of iterative refinement steps
lr = 1e-3
steps = 100000
n_train = 100000
n_test = 1000
batch_size = 128

# Precompute row/col/box indices for all 81 cells
ROW_IDX = torch.tensor([i // 9 for i in range(81)])
COL_IDX = torch.tensor([i % 9 for i in range(81)])
BOX_IDX = torch.tensor([(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)])

class SudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 10 (puzzle) + 9 (current predictions fed back)
        self.input_proj = nn.Linear(10 + 9, d_model)

        # Structured positional embeddings: row, col, box
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
        # x: (batch, 81, 10) - original puzzle encoding
        batch_size = x.size(0)
        device = x.device

        # Get positional embeddings (same for all batches)
        row_idx = ROW_IDX.to(device)
        col_idx = COL_IDX.to(device)
        box_idx = BOX_IDX.to(device)
        pos_embed = self.row_embed(row_idx) + self.col_embed(col_idx) + self.box_embed(box_idx)  # (81, d_model)

        # Initialize predictions as uniform (no info yet)
        preds = torch.zeros(batch_size, 81, 9, device=device)

        all_logits = []
        # Iterative refinement
        for _ in range(n_iterations):
            # Concatenate puzzle encoding with current predictions
            x_in = torch.cat([x, preds], dim=-1)  # (batch, 81, 19)
            h = self.input_proj(x_in)  # (batch, 81, d_model)
            h = h + pos_embed  # add structured positional encoding
            h = self.transformer(h)  # (batch, 81, d_model)
            logits = self.output_head(h)  # (batch, 81, 9)
            preds = F.softmax(logits, dim=-1)  # update predictions for next iteration
            if return_all:
                all_logits.append(logits)

        if return_all:
            return all_logits  # list of (batch, 81, 9) for each iteration
        return logits  # return final logits

def encode_puzzle(puzzle_str):
    """Convert puzzle string to (81, 10) tensor."""
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1  # empty indicator
        else:
            x[i, int(c)] = 1  # one-hot digit (index 1-9)
    return x

def get_targets(puzzle_str, solution_str):
    """Return (hole_indices, target_digits) for empty cells."""
    holes = []
    targets = []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.':
            holes.append(i)
            targets.append(int(s) - 1)  # 0-8
    return torch.tensor(holes), torch.tensor(targets)

df = pd.read_csv("data/sudoku-3m.csv")
easy_df = df[df['difficulty'] == df['difficulty'].min()]
print(f"Puzzles at easiest difficulty: {len(easy_df)} (need {n_train + n_test})")
df = easy_df.head(n_train + n_test)

train_puzzles = df['puzzle'].tolist()[:n_train]
train_solutions = df['solution'].tolist()[:n_train]
test_puzzles = df['puzzle'].tolist()[n_train:]
test_solutions = df['solution'].tolist()[n_train:]

# Override test data in TEST_MODE
if False:
    # Take a real solution and punch N holes at the start
    n_holes = 5
    dummy_solution = train_solutions[0]
    dummy_puzzle = "." * n_holes + dummy_solution[n_holes:]
    assert len(dummy_puzzle) == 81, f"Puzzle length {len(dummy_puzzle)} != 81"
    test_puzzles = [dummy_puzzle]
    test_solutions = [dummy_solution]
    n_test = 1
    print(f"TEST MODE: 1 puzzle with {n_holes} holes")
    print(f"  Puzzle:   {dummy_puzzle[:9]}...")
    print(f"  Solution: {dummy_solution[:9]}...")
    print(f"  Holes at positions 0-{n_holes-1} -> expect '{dummy_solution[:n_holes]}'")

print(f"Train: {len(train_puzzles)}, Test: {len(test_puzzles)} (difficulty={df['difficulty'].iloc[0]})")

# Prepare training data
x_train = torch.stack([encode_puzzle(p) for p in train_puzzles])  # (n_train, 81, 10)
train_holes = [get_targets(p, s) for p, s in zip(train_puzzles, train_solutions)]

# Prepare test data
x_test = torch.stack([encode_puzzle(p) for p in test_puzzles])  # (n_test, 81, 10)
test_holes = [get_targets(p, s) for p, s in zip(test_puzzles, test_solutions)]

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SudokuTransformer().to(device)
model = torch.compile(model)
x_train = x_train.to(device)
x_test = x_test.to(device)

# Move hole tensors to device and precompute counts
train_holes = [(h[0].to(device), h[1].to(device)) for h in train_holes]
train_holes_count = [len(h[0]) for h in train_holes]
test_holes = [(h[0].to(device), h[1].to(device)) for h in test_holes]
test_holes_count = [len(h[0]) for h in test_holes]

# Split params: Muon for ≥2D weights, AdamW for biases/norms/embeddings
muon_params = []
adamw_params = []
for name, param in model.named_parameters():
    if param.ndim >= 2:
        muon_params.append(param)
    else:
        adamw_params.append(param)

optimizer_muon = Muon(muon_params, lr=0.02, momentum=0.95)
optimizer_adamw = torch.optim.AdamW(adamw_params, lr=lr, betas=(0.9, 0.95))

def evaluate_test():
    """Evaluate on test set, return (loss, accuracy, puzzles_solved)."""
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

            # Gather holes for this batch using torch ops
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

            # Check full puzzle solves
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
print(f"\nTraining on {device}...")

# File logging
log_file = open("train_log.txt", "w")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

for step in range(steps):
    model.train()
    optimizer_muon.zero_grad()
    optimizer_adamw.zero_grad()

    # Sample mini-batch
    batch_idx = torch.randperm(n_train)[:batch_size]
    x_batch = x_train[batch_idx]  # (batch_size, 81, 10)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        all_logits = model(x_batch, return_all=True)  # list of (batch_size, 81, 9)

        # Gather holes for this batch using torch ops
        batch_list = batch_idx.tolist()
        hole_c = torch.cat([train_holes[i][0] for i in batch_list])
        targets = torch.cat([train_holes[i][1] for i in batch_list])
        counts = torch.tensor([train_holes_count[i] for i in batch_list], device=device)
        hole_b = torch.repeat_interleave(torch.arange(len(batch_list), device=device), counts)

        # Sum loss over all iterations (intermediate supervision)
        loss = 0
        for logits in all_logits:
            logits_holes = logits[hole_b, hole_c]
            loss = loss + F.cross_entropy(logits_holes, targets)
        loss = loss / len(all_logits)  # average across iterations

    loss.backward()
    optimizer_muon.step()
    optimizer_adamw.step()

    if step % 100 == 0 or step == steps - 1:
        with torch.no_grad():
            # Use final iteration's logits for accuracy
            final_logits_holes = all_logits[-1][hole_b, hole_c]
            preds = final_logits_holes.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()
        train_loss = loss.item()

        # Evaluate on test set every 1000 steps
        if step % 1000 == 0 or step == steps - 1:
            test_loss, test_acc, puzzles_solved = evaluate_test()
            log(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Test Loss: {test_loss:.4f} Acc: {test_acc:.2%} | Solved: {puzzles_solved}/{n_test}")
        else:
            log(f"Step {step:5d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%}")

# Final test evaluation
test_loss, test_acc, puzzles_solved = evaluate_test()
log(f"\nFinal Test: Loss {test_loss:.4f} | Acc {test_acc:.1%} | {puzzles_solved}/{n_test} puzzles solved")
log_file.close()
