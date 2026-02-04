# RRN Experiment: Regular curriculum (easy -> hard) on sudoku-extreme
# BS=1024, 70K steps, AdamW (no SAM)
# Based on sudoku_rrn.py + transformer curriculum experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import random
import os
from checkpoint_utils import find_latest_checkpoint, load_checkpoint, restore_optimizer, save_checkpoint

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "rrn_curriculum_checkpoint_step"

CONFIG = {
    'experiment': 'rrn_exp_curriculum',
    'hidden_size': 128,
    'num_steps': 16,
    'batch_size': 1024,
    'lr': 2e-3,
}

# Hyperparameters
hidden_size = 128
num_steps = 16
lr = 2e-3
total_steps = 70000
batch_size = 1024
train_size = 2700000

# Regular curriculum phases (easy -> hard)
PHASES = [
    (0, 14000, (0, 2), "Phase 1: Easy only (rating 0-2)"),
    (14000, 28000, (0, 10), "Phase 2: Easy-Med (rating 0-10)"),
    (28000, 42000, (0, 50), "Phase 3: Easy-Hard (rating 0-50)"),
    (42000, 70000, (0, 1000), "Phase 4: All (rating 0+)"),
]

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def build_sudoku_edges():
    """Build edge list for Sudoku graph."""
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
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            msg_input = torch.cat([h_src, h_dst], dim=-1)
            messages = self.message_mlp(msg_input)
            aggregated = torch.zeros(batch_size, num_nodes, hidden_size, device=device, dtype=messages.dtype)
            dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_size)
            aggregated.scatter_add_(1, dst_expanded, messages)
            node_input = torch.cat([h, x_embed, aggregated], dim=-1)
            h = h + self.node_mlp(node_input)
            h = self.layer_norm(h)
            if return_all:
                logits = self.output_head(h)
                all_logits.append(logits)

        if return_all:
            return all_logits
        return self.output_head(h)


def encode_puzzle(puzzle_str):
    x = torch.zeros(81, 10)
    for i, c in enumerate(puzzle_str):
        if c == '.':
            x[i, 0] = 1
        else:
            x[i, int(c)] = 1
    return x


def get_targets(puzzle_str, solution_str):
    holes, targets = [], []
    for i, (p, s) in enumerate(zip(puzzle_str, solution_str)):
        if p == '.':
            holes.append(i)
            targets.append(int(s) - 1)
    return torch.tensor(holes), torch.tensor(targets)


def train(output_dir="."):
    device = torch.device("cuda")
    edge_index = build_sudoku_edges().to(device)

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

    # Build phase buckets for regular curriculum (rating ranges)
    phase_buckets = {}
    for start, end, rating_range, name in PHASES:
        min_rating, max_rating = rating_range
        buckets_for_phase = [k for k in train_data.keys() if k[0] >= min_rating and k[1] <= max_rating]
        total = sum(train_data[k]['size'] for k in buckets_for_phase)
        phase_buckets[rating_range] = buckets_for_phase
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

    model = SudokuRRN(edge_index).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,} (RRN baseline: ~194K)")

    checkpoint_path, start_step = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    checkpoint_data = None
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, CONFIG)
        start_step = checkpoint_data['step']
        print(f"Loaded model weights from step {start_step}")

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if checkpoint_data:
        restore_optimizer(optimizer, checkpoint_data, device)
        print(f"Resumed from step {start_step}")

    print(f"\nExperiment: RRN Regular Curriculum (easy->hard)")
    print(f"Architecture: hidden_size={hidden_size}, num_steps={num_steps}")
    print(f"Batch size: {batch_size}, lr: {lr}")
    print(f"Output directory: {output_dir}")

    log_path = os.path.join(output_dir, "rrn_exp_curriculum.log")
    log_file = open(log_path, "a")
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs", "rrn_curriculum"))

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def get_phase(step):
        for start, end, rating_range, name in PHASES:
            if start <= step < end:
                return phase_buckets[rating_range], name
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

    def evaluate_all():
        model.eval()
        results = {}
        total_solved, total_puzzles = 0, 0
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
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt")
        save_checkpoint(path, step, model, optimizer, CONFIG)
        print(f"Checkpoint saved: {path}")

    current_phase_name = None
    current_buckets = None

    for step in range(start_step, total_steps):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == total_steps - 1:
            with torch.no_grad():
                final_logits_holes = all_logits[-1][hole_b, hole_c]
                preds = final_logits_holes.argmax(dim=-1)
                train_acc = (preds == targets).float().mean().item()

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", train_acc * 100, step)

            if step % 5000 == 0 or step == total_steps - 1:
                results = evaluate_all()
                total_r = results.pop('_total')
                test_acc = 100 * total_r['solved'] / total_r['total']
                writer.add_scalar("Accuracy/test", test_acc, step)
                log(f"Step {step:5d} | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({test_acc:.1f}%)")
                do_save_checkpoint(step)
            else:
                log(f"Step {step:5d} | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

    log("\n" + "="*60)
    log("FINAL RESULTS - RRN Regular Curriculum")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")

    final_path = os.path.join(output_dir, "model_rrn_curriculum.pt")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
    torch.save(state_dict, final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    writer.close()


if __name__ == "__main__":
    train()
