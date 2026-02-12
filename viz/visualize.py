# Visualization of attention patterns, confidence evolution, and head specialization
# for the iterative sudoku transformer.
#
# Usage: python viz/visualize.py model_bs2048_baseline.pt --exp iters.exp_bs2048_baseline
#
# Generates figures in viz/output/

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import os
import sys
import argparse
import importlib
from datasets import load_dataset
import random

random.seed(42)
torch.manual_seed(42)


def get_attention_weights(model, x, n_iters, exp_mod):
    """Run forward pass capturing attention weights at every layer/head/iteration."""
    device = x.device
    rope_cos = exp_mod.ROPE_COS.to(device)
    rope_sin = exp_mod.ROPE_SIN.to(device)

    batch_size = x.size(0)
    h_prev = model.initial_encoder(x)
    preds = torch.zeros(batch_size, 81, 9, device=device)

    all_attention = []  # [iter][layer] -> (B, H, 81, 81)
    all_logits = []     # [iter] -> (B, 81, 9)

    for iter_idx in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        iter_attn = []

        for layer in model.layers:
            # Manually compute attention weights instead of using SDPA
            h_norm = layer.norm1(h)
            B, L, D = h_norm.shape
            n_heads = layer.n_heads
            head_dim = layer.head_dim

            q = layer.q_proj(h_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = layer.k_proj(h_norm).view(B, L, n_heads, head_dim).transpose(1, 2)
            v = layer.v_proj(h_norm).view(B, L, n_heads, head_dim).transpose(1, 2)

            q = exp_mod.apply_rope(q, rope_cos, rope_sin)
            k = exp_mod.apply_rope(k, rope_cos, rope_sin)

            # Compute attention weights explicitly
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)  # (B, H, 81, 81)
            iter_attn.append(attn_weights.detach().cpu())

            # Still compute the actual forward pass for correct h_prev
            attn_out = torch.matmul(attn_weights.to(v.dtype), v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
            h = h + layer.out_proj(attn_out)

            h2 = layer.norm2(h)
            h2 = layer.linear2(F.relu(layer.linear1(h2)))
            h = h + h2

        h_prev = h
        logits = model.output_head(h)
        preds = F.softmax(logits, dim=-1)
        all_logits.append(logits.detach().cpu())
        all_attention.append(iter_attn)

    return all_attention, all_logits


def cell_to_rc(cell_idx):
    return cell_idx // 9, cell_idx % 9


def draw_sudoku_grid(ax):
    """Draw 3x3 box borders on a 9x9 grid."""
    for i in range(0, 10, 3):
        ax.axhline(i - 0.5, color='black', linewidth=2)
        ax.axvline(i - 0.5, color='black', linewidth=2)


def plot_attention_for_cell(all_attention, query_cell, puzzle_str, solution_str,
                            n_layers, n_heads, iteration, output_dir, puzzle_idx):
    """Plot what a specific cell attends to, for each head and layer at a given iteration."""
    qr, qc = cell_to_rc(query_cell)
    iter_attn = all_attention[iteration]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_heads == 1:
        axes = axes[:, np.newaxis]

    for layer_idx in range(n_layers):
        attn = iter_attn[layer_idx][0]  # (H, 81, 81), first puzzle in batch
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            weights = attn[head_idx, query_cell, :].numpy().reshape(9, 9)
            im = ax.imshow(weights, cmap='hot', vmin=0, interpolation='nearest')
            draw_sudoku_grid(ax)

            # Mark query cell
            ax.add_patch(Rectangle((qc - 0.5, qr - 0.5), 1, 1,
                                   fill=False, edgecolor='cyan', linewidth=2))

            # Show given digits
            for i in range(81):
                r, c = cell_to_rc(i)
                if puzzle_str[i] != '.':
                    ax.text(c, r, puzzle_str[i], ha='center', va='center',
                            fontsize=6, color='white', fontweight='bold')

            ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    cell_char = solution_str[query_cell] if solution_str else '?'
    fig.suptitle(f'Attention from cell ({qr},{qc}) [answer={cell_char}], iter {iteration}',
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f'P{puzzle_idx}_attn_cell{query_cell}_iter{iteration}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_confidence_evolution(all_logits, puzzle_str, solution_str, output_dir, puzzle_idx,
                              n_iters_to_show=None):
    """Heatmap of per-cell confidence (max softmax) across iterations."""
    n_iters = len(all_logits)
    if n_iters_to_show is None:
        # Show a subset of iterations for readability
        if n_iters <= 16:
            iter_indices = list(range(n_iters))
        else:
            iter_indices = [0, 1, 2, 3, 4, 7, 10, 15] + list(range(16, n_iters, max(1, (n_iters - 16) // 4)))
            iter_indices = sorted(set(i for i in iter_indices if i < n_iters))
    else:
        iter_indices = list(range(min(n_iters_to_show, n_iters)))

    n_show = len(iter_indices)
    cols = min(8, n_show)
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    # Build target for correctness checking
    target = [int(solution_str[i]) - 1 for i in range(81)]
    empty_mask = [puzzle_str[i] == '.' for i in range(81)]

    for plot_idx, iter_idx in enumerate(iter_indices):
        r, c = plot_idx // cols, plot_idx % cols
        ax = axes[r, c]

        logits = all_logits[iter_idx][0]  # (81, 9)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.numpy().reshape(9, 9)
        predictions = logits.argmax(dim=-1).numpy()

        # Color by correctness: green=correct, red=wrong, gray=given
        color_grid = np.zeros((9, 9, 3))
        for i in range(81):
            row, col = cell_to_rc(i)
            if not empty_mask[i]:
                color_grid[row, col] = [0.85, 0.85, 0.85]  # given
            elif predictions[i] == target[i]:
                color_grid[row, col] = [0.2, 0.7, 0.2]  # correct - green
            else:
                color_grid[row, col] = [0.8, 0.2, 0.2]  # wrong - red
            # Modulate by confidence
            if empty_mask[i]:
                alpha = 0.3 + 0.7 * confidence[row, col]
                color_grid[row, col] *= alpha

        ax.imshow(color_grid, interpolation='nearest')
        draw_sudoku_grid(ax)

        # Show predictions
        for i in range(81):
            row, col = cell_to_rc(i)
            if puzzle_str[i] != '.':
                ax.text(col, row, puzzle_str[i], ha='center', va='center',
                        fontsize=6, color='black')
            else:
                pred_digit = str(predictions[i] + 1)
                ax.text(col, row, pred_digit, ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold')

        # Count correct
        n_correct = sum(1 for i in range(81) if empty_mask[i] and predictions[i] == target[i])
        n_empty = sum(empty_mask)
        ax.set_title(f'Iter {iter_idx} ({n_correct}/{n_empty})', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for plot_idx in range(n_show, rows * cols):
        r, c = plot_idx // cols, plot_idx % cols
        axes[r, c].set_visible(False)

    fig.suptitle(f'Confidence evolution (green=correct, red=wrong, gray=given)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f'P{puzzle_idx}_confidence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_entropy_evolution(all_logits, puzzle_str, output_dir, puzzle_idx):
    """Line plot showing average entropy of empty cells across iterations."""
    empty_mask = torch.tensor([puzzle_str[i] == '.' for i in range(81)])
    n_iters = len(all_logits)

    entropies = []
    for iter_idx in range(n_iters):
        logits = all_logits[iter_idx][0]  # (81, 9)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (81,)
        avg_entropy = entropy[empty_mask].mean().item()
        entropies.append(avg_entropy)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(n_iters), entropies, 'b-o', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average entropy (empty cells)')
    ax.set_title(f'Entropy evolution (max possible = {np.log(9):.2f})')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'P{puzzle_idx}_entropy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_head_specialization(all_attention, puzzles, n_layers, n_heads, output_dir):
    """Average attention patterns across many puzzles to reveal head specialization.
    For each head, show the average attention pattern as a function of
    row/col distance between query and key cells."""
    n_puzzles = len(puzzles)
    n_iters = len(all_attention[0])

    # Use last iteration's attention (most refined)
    last_iter = n_iters - 1

    # Compute average attention as function of (row_diff, col_diff)
    # This reveals if heads specialize in rows, columns, or boxes
    for layer_idx in range(n_layers):
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 3.5))
        if n_heads == 1:
            axes = [axes]

        for head_idx in range(n_heads):
            # Accumulate attention by relative position
            pos_attn = np.zeros((17, 17))  # -8 to +8 for both row and col diff
            pos_count = np.zeros((17, 17))

            for p_idx in range(n_puzzles):
                attn = all_attention[p_idx][last_iter][layer_idx][0]  # (H, 81, 81)
                head_attn = attn[head_idx].numpy()  # (81, 81)

                for q in range(81):
                    qr, qc = cell_to_rc(q)
                    for k in range(81):
                        kr, kc = cell_to_rc(k)
                        dr = kr - qr + 8  # shift to 0-16 range
                        dc = kc - qc + 8
                        pos_attn[dr, dc] += head_attn[q, k]
                        pos_count[dr, dc] += 1

            pos_count[pos_count == 0] = 1
            avg_attn = pos_attn / pos_count

            ax = axes[head_idx]
            im = ax.imshow(avg_attn, cmap='hot', interpolation='nearest',
                           extent=[-8.5, 8.5, 8.5, -8.5])
            ax.set_xlabel('Col offset')
            ax.set_ylabel('Row offset')
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.axhline(0, color='cyan', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='cyan', linewidth=0.5, alpha=0.5)
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f'Layer {layer_idx}: Average attention by relative position (iter {last_iter})',
                     fontsize=11)
        plt.tight_layout()
        path = os.path.join(output_dir, f'head_specialization_layer{layer_idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_attention_across_iterations(all_attention, query_cell, puzzle_str, solution_str,
                                     n_layers, n_heads, output_dir, puzzle_idx):
    """Show how attention pattern for a specific cell evolves across iterations."""
    n_iters = len(all_attention)
    # Pick subset of iterations
    if n_iters <= 8:
        iter_indices = list(range(n_iters))
    else:
        iter_indices = [0, 1, 3, 7, 11, 15]
        iter_indices = [i for i in iter_indices if i < n_iters]

    qr, qc = cell_to_rc(query_cell)

    # Show for layer 0 head 0, and last layer last head (first and last)
    head_configs = [(0, 0), (n_layers - 1, n_heads - 1)]

    for layer_idx, head_idx in head_configs:
        fig, axes = plt.subplots(1, len(iter_indices), figsize=(2.5 * len(iter_indices), 3))
        if len(iter_indices) == 1:
            axes = [axes]

        for plot_idx, iter_idx in enumerate(iter_indices):
            ax = axes[plot_idx]
            attn = all_attention[iter_idx][layer_idx][0]  # (H, 81, 81)
            weights = attn[head_idx, query_cell, :].numpy().reshape(9, 9)
            ax.imshow(weights, cmap='hot', vmin=0, interpolation='nearest')
            draw_sudoku_grid(ax)
            ax.add_patch(Rectangle((qc - 0.5, qr - 0.5), 1, 1,
                                   fill=False, edgecolor='cyan', linewidth=2))
            ax.set_title(f'Iter {iter_idx}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        cell_char = solution_str[query_cell]
        fig.suptitle(f'L{layer_idx}H{head_idx}: cell ({qr},{qc}) [ans={cell_char}] across iters',
                     fontsize=10)
        plt.tight_layout()
        path = os.path.join(output_dir,
                            f'P{puzzle_idx}_attn_evolution_cell{query_cell}_L{layer_idx}H{head_idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to model .pt file')
    parser.add_argument('--exp', default='iters.exp_bs2048_baseline')
    parser.add_argument('--n-iters', type=int, default=16, help='Number of iterations to run')
    parser.add_argument('--n-puzzles', type=int, default=20, help='Puzzles for head specialization')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    exp_mod = importlib.import_module(args.exp)
    device = torch.device(args.device)
    model = exp_mod.SudokuTransformer().to(device)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    n_layers = exp_mod.n_layers
    n_heads = exp_mod.n_heads
    print(f'Model: d_model={exp_mod.d_model}, n_heads={n_heads}, n_layers={n_layers}')
    print(f'Iterations: {args.n_iters}')

    # Load test data - pick representative puzzles
    print('Loading test data...')
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")

    # Pick one easy, one medium, one hard puzzle (that the model solves)
    easy = [i for i in range(min(1000, len(test_dataset))) if test_dataset[i]['rating'] == 0]
    medium = [i for i in range(min(5000, len(test_dataset))) if 3 <= test_dataset[i]['rating'] <= 10]
    hard = [i for i in range(min(10000, len(test_dataset))) if test_dataset[i]['rating'] >= 51]

    selected = []
    if easy:
        selected.append(('easy', easy[0]))
    if medium:
        selected.append(('medium', medium[0]))
    if hard:
        selected.append(('hard', hard[0]))

    # Find puzzles the model gets WRONG at 16 iterations
    print('Scanning for failed puzzles...')
    scan_size = 200
    scan_indices = random.sample(range(min(10000, len(test_dataset))), scan_size)
    scan_puzzles = [test_dataset[i]['question'] for i in scan_indices]
    scan_solutions = [test_dataset[i]['answer'] for i in scan_indices]
    scan_x = exp_mod.encode_puzzles(scan_puzzles).to(device)

    with torch.no_grad():
        logits = model(scan_x)

    # Check which puzzles are wrong
    final_preds = logits.argmax(dim=-1).cpu()  # (scan_size, 81)
    failed_indices = []
    for i in range(scan_size):
        sol = scan_solutions[i]
        target = torch.tensor([int(sol[c]) - 1 for c in range(81)])
        mask = torch.tensor([scan_puzzles[i][c] == '.' for c in range(81)])
        if not (final_preds[i][mask] == target[mask]).all():
            failed_indices.append(i)

    if failed_indices:
        # Pick up to 2 failed puzzles with different error counts
        fail_errors = []
        for i in failed_indices:
            sol = scan_solutions[i]
            target = torch.tensor([int(sol[c]) - 1 for c in range(81)])
            mask = torch.tensor([scan_puzzles[i][c] == '.' for c in range(81)])
            n_wrong = (final_preds[i][mask] != target[mask]).sum().item()
            fail_errors.append((i, n_wrong))
        fail_errors.sort(key=lambda x: x[1])
        # Pick one with few errors (almost solved) and one with many
        picks = [fail_errors[0]]
        if len(fail_errors) > 1:
            picks.append(fail_errors[-1])
        for scan_i, n_wrong in picks:
            ds_idx = scan_indices[scan_i]
            rating = test_dataset[ds_idx]['rating']
            selected.append((f'FAIL({n_wrong}wrong,r={rating})', ds_idx))
        print(f'  Found {len(failed_indices)}/{scan_size} failed puzzles, picked {len(picks)}')
    else:
        print(f'  Model solved all {scan_size} scanned puzzles!')

    print(f'\nSelected puzzles: {[(name, idx) for name, idx in selected]}')

    # === Per-puzzle visualizations ===
    for puzzle_idx, (difficulty, ds_idx) in enumerate(selected):
        puzzle_str = test_dataset[ds_idx]['question']
        solution_str = test_dataset[ds_idx]['answer']
        rating = test_dataset[ds_idx]['rating']
        n_empty = puzzle_str.count('.')

        # Use more iterations for failed puzzles to see phase transition
        is_fail = difficulty.startswith('FAIL')
        n_iters = max(args.n_iters * 4, 64) if is_fail else args.n_iters
        print(f'\n--- Puzzle {puzzle_idx} ({difficulty}, rating={rating}, {n_empty} empty cells, {n_iters} iters) ---')

        x = exp_mod.encode_puzzles([puzzle_str]).to(device)

        with torch.no_grad():
            all_attention, all_logits = get_attention_weights(model, x, n_iters, exp_mod)

        # 1. Confidence evolution across iterations
        print('  Plotting confidence evolution...')
        plot_confidence_evolution(all_logits, puzzle_str, solution_str, output_dir, puzzle_idx)

        # 2. Entropy evolution
        print('  Plotting entropy evolution...')
        plot_entropy_evolution(all_logits, puzzle_str, output_dir, puzzle_idx)

        # 3. Attention maps for a specific empty cell
        empty_cells = [i for i in range(81) if puzzle_str[i] == '.']
        # Pick a cell near the center for interesting patterns
        center_cells = [i for i in empty_cells if 2 <= i // 9 <= 6 and 2 <= i % 9 <= 6]
        query_cell = center_cells[0] if center_cells else empty_cells[0]
        print(f'  Plotting attention for cell {query_cell} ({cell_to_rc(query_cell)})...')

        # Attention at last iteration
        plot_attention_for_cell(all_attention, query_cell, puzzle_str, solution_str,
                                n_layers, n_heads, n_iters - 1, output_dir, puzzle_idx)

        # 4. Attention evolution across iterations for that cell
        print('  Plotting attention evolution across iterations...')
        plot_attention_across_iterations(all_attention, query_cell, puzzle_str, solution_str,
                                         n_layers, n_heads, output_dir, puzzle_idx)

    # === Head specialization (averaged across many puzzles) ===
    print(f'\n--- Head specialization (averaging over {args.n_puzzles} puzzles) ---')
    spec_indices = random.sample(range(min(5000, len(test_dataset))), args.n_puzzles)
    spec_puzzles = [test_dataset[i]['question'] for i in spec_indices]

    all_puzzle_attentions = []
    for i, puzzle_str in enumerate(spec_puzzles):
        x = exp_mod.encode_puzzles([puzzle_str]).to(device)
        with torch.no_grad():
            attn, _ = get_attention_weights(model, x, args.n_iters, exp_mod)
        all_puzzle_attentions.append(attn)
        if (i + 1) % 5 == 0:
            print(f'  Processed {i + 1}/{args.n_puzzles} puzzles')

    plot_head_specialization(all_puzzle_attentions, spec_puzzles, n_layers, n_heads, output_dir)

    print(f'\nAll figures saved to {output_dir}/')


if __name__ == '__main__':
    main()
