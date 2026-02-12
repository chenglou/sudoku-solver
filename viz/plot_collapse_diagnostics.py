# Comparative diagnostics: why do some models collapse at high iteration counts?
# Runs forward passes on multiple models, tracking hidden state norms and
# prediction stability across iterations.
#
# Usage (local):  python viz/plot_collapse_diagnostics.py model1.pt model2.pt --exps mod1 mod2
# Usage (Modal):  modal run viz/modal_viz.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib
import argparse
from datasets import load_dataset

torch.set_float32_matmul_precision('high')


def run_diagnostics(model, exp_mod, x_batch, n_iters, device):
    """Run forward pass tracking hidden state norms and prediction changes."""
    batch_size = x_batch.size(0)
    rope_cos = exp_mod.ROPE_COS.to(device)
    rope_sin = exp_mod.ROPE_SIN.to(device)

    h_prev = model.initial_encoder(x_batch)
    preds = torch.zeros(batch_size, 81, 9, device=device)

    h_norms = []       # ||h|| at each iteration
    pred_deltas = []   # ||pred_t - pred_{t-1}|| at each iteration
    logit_norms = []   # ||logits|| at each iteration
    prev_preds = None

    for iter_idx in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        logits = model.output_head(h)
        preds = F.softmax(logits, dim=-1)

        # Track hidden state norm (avg over batch and cells)
        h_norm = h.norm(dim=-1).mean().item()  # avg ||h|| per cell
        h_norms.append(h_norm)

        # Track logit norm
        logit_norm = logits.norm(dim=-1).mean().item()
        logit_norms.append(logit_norm)

        # Track prediction change
        if prev_preds is not None:
            delta = (preds - prev_preds).norm(dim=-1).mean().item()
            pred_deltas.append(delta)
        else:
            pred_deltas.append(float('nan'))
        prev_preds = preds.clone()

    return {
        'h_norms': h_norms,
        'pred_deltas': pred_deltas,
        'logit_norms': logit_norms,
    }


def run_accuracy_per_iter(model, exp_mod, x_batch, solutions, empty_masks, n_iters, device):
    """Track per-iteration accuracy and cell flip counts."""
    batch_size = x_batch.size(0)
    rope_cos = exp_mod.ROPE_COS.to(device)
    rope_sin = exp_mod.ROPE_SIN.to(device)

    h_prev = model.initial_encoder(x_batch)
    preds_soft = torch.zeros(batch_size, 81, 9, device=device)

    accuracies = []
    correct_to_wrong = []  # cells that were correct, now wrong
    wrong_to_correct = []  # cells that were wrong, now correct
    prev_correct = None

    for iter_idx in range(n_iters):
        h = h_prev + model.pred_proj(preds_soft)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        logits = model.output_head(h)
        preds_soft = F.softmax(logits, dim=-1)

        preds = logits.argmax(dim=-1).cpu()  # (B, 81)
        correct = (preds == solutions) & empty_masks  # (B, 81)

        acc = correct.sum().float() / empty_masks.sum().float() * 100
        accuracies.append(acc.item())

        if prev_correct is not None:
            flipped_wrong = (prev_correct & ~correct & empty_masks).sum().item()
            flipped_right = (~prev_correct & correct & empty_masks).sum().item()
            correct_to_wrong.append(flipped_wrong)
            wrong_to_correct.append(flipped_right)
        else:
            correct_to_wrong.append(0)
            wrong_to_correct.append(0)

        prev_correct = correct.clone()

    return {
        'accuracies': accuracies,
        'correct_to_wrong': correct_to_wrong,
        'wrong_to_correct': wrong_to_correct,
    }


def load_test_data(exp_mod, n_puzzles=500, device='cuda'):
    """Load a fixed set of test puzzles."""
    dataset = load_dataset("sapientinc/sudoku-extreme", split="test")

    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), n_puzzles)

    puzzles = [dataset[i]['question'] for i in indices]
    solutions_str = [dataset[i]['answer'] for i in indices]

    x = exp_mod.encode_puzzles(puzzles).to(device)

    solutions = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in solutions_str])
    empty_masks = torch.tensor([[p[j] == '.' for j in range(81)] for p in puzzles])

    return x, solutions, empty_masks


def plot_diagnostics(all_results, output_dir):
    """Generate comparison plots from diagnostic results."""
    n_iters = len(next(iter(all_results.values()))['h_norms'])
    iters = list(range(n_iters))

    # 1. Hidden state norms
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in all_results.items():
        ax.plot(iters, r['h_norms'], label=name, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean ||h|| per cell')
    ax.set_title('Hidden State Norm Across Iterations')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diag_h_norms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # 2. Prediction deltas (convergence)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in all_results.items():
        ax.plot(iters[1:], r['pred_deltas'][1:], label=name, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean ||pred_t - pred_{t-1}||')
    ax.set_title('Prediction Change Between Consecutive Iterations')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diag_pred_deltas.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # 3. Logit norms
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in all_results.items():
        ax.plot(iters, r['logit_norms'], label=name, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean ||logits|| per cell')
    ax.set_title('Logit Norm Across Iterations')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diag_logit_norms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_flip_analysis(all_flip_results, output_dir):
    """Plot cell flip analysis across models."""
    n_iters = len(next(iter(all_flip_results.values()))['accuracies'])
    iters = list(range(n_iters))

    # 1. Accuracy curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in all_flip_results.items():
        ax.plot(iters, r['accuracies'], label=name, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cell-level accuracy (%)')
    ax.set_title('Per-Cell Accuracy Across Iterations')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diag_accuracy_per_iter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # 2. Cell flips (correct→wrong vs wrong→correct)
    fig, axes = plt.subplots(1, len(all_flip_results), figsize=(5 * len(all_flip_results), 4),
                              squeeze=False)
    for idx, (name, r) in enumerate(all_flip_results.items()):
        ax = axes[0, idx]
        ax.plot(iters[1:], r['wrong_to_correct'][1:], label='wrong→correct', color='green', linewidth=1)
        ax.plot(iters[1:], r['correct_to_wrong'][1:], label='correct→wrong', color='red', linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cells flipped')
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Cell Flips Per Iteration', fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, 'diag_cell_flips.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def analyze_models(model_configs, n_iters=256, n_puzzles=500, device='cuda', output_dir='.'):
    """
    model_configs: list of (name, model_path, exp_module_name)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load test data using first model's exp module (they all use same data)
    first_exp = importlib.import_module(model_configs[0][2])
    x, solutions, empty_masks = load_test_data(first_exp, n_puzzles, device)

    all_diag = {}
    all_flips = {}

    for name, model_path, exp_name in model_configs:
        print(f'\n=== {name} ===')
        exp_mod = importlib.import_module(exp_name)
        model = exp_mod.SudokuTransformer().to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        model.eval()

        x_this = x

        print(f'  Running {n_iters} iterations on {n_puzzles} puzzles...')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Process in batches
            batch_size = 128
            diag_accum = {'h_norms': [], 'pred_deltas': [], 'logit_norms': []}
            flip_accum = {'accuracies': [], 'correct_to_wrong': [], 'wrong_to_correct': []}

            for start in range(0, n_puzzles, batch_size):
                end = min(start + batch_size, n_puzzles)
                batch_x = x_this[start:end]
                batch_sol = solutions[start:end]
                batch_mask = empty_masks[start:end]

                diag = run_diagnostics(model, exp_mod, batch_x, n_iters, device)
                flips = run_accuracy_per_iter(model, exp_mod, batch_x, batch_sol, batch_mask,
                                              n_iters, device)

                diag_accum['h_norms'].append(diag['h_norms'])
                diag_accum['pred_deltas'].append(diag['pred_deltas'])
                diag_accum['logit_norms'].append(diag['logit_norms'])
                flip_accum['accuracies'].append(np.array(flips['accuracies']) * (end - start))
                flip_accum['correct_to_wrong'].append(flips['correct_to_wrong'])
                flip_accum['wrong_to_correct'].append(flips['wrong_to_correct'])

            # Average across batches
            n_batches = len(diag_accum['h_norms'])
            all_diag[name] = {
                'h_norms': [sum(diag_accum['h_norms'][b][i] for b in range(n_batches)) / n_batches
                            for i in range(n_iters)],
                'pred_deltas': [sum(diag_accum['pred_deltas'][b][i] for b in range(n_batches)) / n_batches
                                for i in range(n_iters)],
                'logit_norms': [sum(diag_accum['logit_norms'][b][i] for b in range(n_batches)) / n_batches
                                for i in range(n_iters)],
            }
            all_flips[name] = {
                'accuracies': [sum(flip_accum['accuracies'][b][i] for b in range(n_batches)) / n_puzzles
                               for i in range(n_iters)],
                'correct_to_wrong': [sum(flip_accum['correct_to_wrong'][b][i] for b in range(n_batches))
                                     for i in range(n_iters)],
                'wrong_to_correct': [sum(flip_accum['wrong_to_correct'][b][i] for b in range(n_batches))
                                     for i in range(n_iters)],
            }

        del model
        torch.cuda.empty_cache()
        print(f'  Done.')

    print('\nGenerating plots...')
    plot_diagnostics(all_diag, output_dir)
    plot_flip_analysis(all_flips, output_dir)
    print('All plots saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', help='model .pt paths')
    parser.add_argument('--exps', nargs='+', required=True, help='experiment module names')
    parser.add_argument('--names', nargs='+', help='display names')
    parser.add_argument('--n-iters', type=int, default=256)
    parser.add_argument('--n-puzzles', type=int, default=500)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='viz/output')
    args = parser.parse_args()

    names = args.names or [os.path.basename(m).replace('.pt', '') for m in args.models]
    configs = list(zip(names, args.models, args.exps))
    analyze_models(configs, args.n_iters, args.n_puzzles, args.device, args.output_dir)
