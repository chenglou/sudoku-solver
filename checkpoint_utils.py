import os
import re
import glob
import torch


def find_latest_checkpoint(output_dir, checkpoint_prefix):
    """Find the latest checkpoint file and return (path, step) or (None, 0)."""
    pattern = os.path.join(output_dir, f"{checkpoint_prefix}_step*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    def get_step(path):
        match = re.search(r'step(\d+)\.pt$', path)
        return int(match.group(1)) if match else 0
    latest = max(checkpoints, key=get_step)
    return latest, get_step(latest)


def load_checkpoint(path, model, config):
    """Load checkpoint and verify config matches. Returns checkpoint dict."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Verify config matches
    saved_config = checkpoint.get('config', {})
    for key, value in config.items():
        saved_value = saved_config.get(key)
        if saved_value != value:
            raise ValueError(
                f"Config mismatch! {key}: saved={saved_value}, current={value}. "
                f"Use a fresh output_dir or delete old checkpoints."
            )

    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
