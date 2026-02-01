# Convert existing experiment logs to TensorBoard format
# Usage: python logs_to_tensorboard.py
# Then: tensorboard --logdir runs/

import os
import re
from torch.utils.tensorboard import SummaryWriter

LOG_FILES = [
    ("logs/exp_extreme_baseline.log", "baseline_bs512"),
    ("logs/exp_scale_batch.log", "bs2048"),
    ("logs/exp_scale_batch_4k.log", "bs4096_70k"),
    ("logs/exp_scale_batch_4k_v2.log", "bs4096_10k_reverse"),
    ("logs/exp_scale_batch_4k_curriculum.log", "bs4096_10k_regular"),
    ("logs/exp_scale_wide.log", "scale_wide"),
    ("logs/exp_scale_up_big_gpu.log", "scale_up"),
    ("logs/exp_warmup.log", "warmup"),
    ("logs/exp_cosine.log", "cosine"),  # NEW SOTA 84.0%
    ("logs/exp_ema.log", "ema"),
]

# Pattern for log lines with eval results
# Step  5000 | Loss: 0.7533 Acc: 79.12% | 0: 4097/5000 | ... | Total: 13723/25000 (54.9%)
# Also handles: Step     0 | LR: 7.50e-07 | Loss: 17.7026 Acc: 11.10% | ... | Total: 0/25000 (0.0%)
EVAL_PATTERN = re.compile(
    r"Step\s+(\d+)\s+\|.*Loss:\s+([\d.]+)\s+Acc:\s+([\d.]+)%.*Total:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)"
)

# Pattern for regular training lines (no Total:)
# Step  5100 | Loss: 0.7234 Acc: 78.45%
# Also handles: Step   100 | LR: 7.57e-05 | Loss: 5.1433 Acc: 11.26%
TRAIN_PATTERN = re.compile(
    r"Step\s+(\d+)\s+\|.*Loss:\s+([\d.]+)\s+Acc:\s+([\d.]+)%"
)


def parse_log(filepath):
    """Parse a log file and extract metrics."""
    metrics = []

    with open(filepath) as f:
        for line in f:
            # Try eval pattern first (has Total:)
            match = EVAL_PATTERN.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                train_acc = float(match.group(3))
                test_acc = float(match.group(6))
                metrics.append({
                    'step': step,
                    'loss': loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                })
                continue

            # Try regular training pattern
            match = TRAIN_PATTERN.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                train_acc = float(match.group(3))
                metrics.append({
                    'step': step,
                    'loss': loss,
                    'train_acc': train_acc,
                    'test_acc': None,
                })

    return metrics


def write_to_tensorboard(metrics, run_name):
    """Write metrics to TensorBoard."""
    writer = SummaryWriter(f"runs/{run_name}")

    for m in metrics:
        step = m['step']
        writer.add_scalar("Loss/train", m['loss'], step)
        writer.add_scalar("Accuracy/train", m['train_acc'], step)
        if m['test_acc'] is not None:
            writer.add_scalar("Accuracy/test", m['test_acc'], step)

    writer.close()
    print(f"  Wrote {len(metrics)} entries to runs/{run_name}")


def main():
    os.makedirs("runs", exist_ok=True)

    for log_file, run_name in LOG_FILES:
        if not os.path.exists(log_file):
            print(f"Skipping {log_file} (not found)")
            continue

        print(f"Processing {log_file}...")
        metrics = parse_log(log_file)
        write_to_tensorboard(metrics, run_name)

    print("\nDone! Run: tensorboard --logdir runs/")


if __name__ == "__main__":
    main()
