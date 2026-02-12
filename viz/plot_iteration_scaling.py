# Plot accuracy vs test-time iterations for multiple models.
# Shows which models scale stably and where others collapse.
#
# Usage: python viz/plot_iteration_scaling.py

import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# Data from EXPERIMENTS_ITERS.md eval results
models = {
    'LR=2e-3 (SOTA)': {
        'iters': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'acc': [81.8, 88.5, 92.5, 95.3, 97.3, 98.5, 98.9, 98.8],
        'style': {'color': '#2ecc71', 'linewidth': 2.5, 'marker': 'o'},
    },
    'LR=1.5e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'acc': [81.4, 88.1, 92.4, 94.9, 96.6, 97.5, 98.1, 98.2],
        'style': {'color': '#27ae60', 'linewidth': 1.5, 'marker': 's', 'linestyle': '--'},
    },
    'LR=2.5e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024],
        'acc': [81.8, 87.7, 90.7, 84.1, 50.5, 4.5, 0.1],
        'style': {'color': '#e74c3c', 'linewidth': 1.5, 'marker': '^'},
    },
    'LR=3e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024],
        'acc': [82.1, 88.5, 39.9, 7.1, 1.7, 0.9, 0.4],
        'style': {'color': '#c0392b', 'linewidth': 1.5, 'marker': 'v'},
    },
    'LR=1e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'acc': [80.9, 87.2, 90.9, 89.8, 80.2, 50.2, 20.8, 5.3],
        'style': {'color': '#e67e22', 'linewidth': 1.5, 'marker': 'D'},
    },
}

models_size = {
    'd=128 LR=2e-3 (SOTA)': {
        'iters': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'acc': [81.8, 88.5, 92.5, 95.3, 97.3, 98.5, 98.9, 98.8],
        'style': {'color': '#2ecc71', 'linewidth': 2.5, 'marker': 'o'},
    },
    'd=192 LR=2e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024],
        'acc': [84.4, 90.8, 94.3, 85.9, 23.3, 6.9, 3.5],
        'style': {'color': '#9b59b6', 'linewidth': 1.5, 'marker': '^'},
    },
    'd=192 LR=1.5e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024],
        'acc': [84.7, 91.3, 64.2, 28.2, 11.5, 4.5, 1.1],
        'style': {'color': '#8e44ad', 'linewidth': 1.5, 'marker': 'v'},
    },
    'd=192 LR=1e-3': {
        'iters': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'acc': [84.1, 90.8, 94.5, 94.7, 8.2, 0.9, 0.1, 0.0],
        'style': {'color': '#7d3c98', 'linewidth': 1.5, 'marker': 'D'},
    },
    'd=96': {
        'iters': [16, 32, 64, 128, 256, 512, 1024, 2048],
        'acc': [76.8, 83.0, 86.5, 87.8, 87.7, 87.4, 86.4, 73.2],
        'style': {'color': '#f39c12', 'linewidth': 1.5, 'marker': 's', 'linestyle': '--'},
    },
}


def make_plot(data, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, d in data.items():
        ax.plot(d['iters'], d['acc'], label=name, markersize=5, **d['style'])

    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(set(i for d in data.values() for i in d['iters'])))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel('Test-time iterations', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


make_plot(models, 'Iteration Scaling by Learning Rate (d=128, BS=2048)', 'iter_scaling_lr.png')
make_plot(models_size, 'Iteration Scaling by Model Size', 'iter_scaling_size.png')
