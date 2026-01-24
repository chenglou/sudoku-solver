# Checkpoint utilities for Modal-safe training resumption
# Load checkpoint BEFORE torch.compile() to avoid _orig_mod. key mismatch

import os
import glob
import torch


def find_latest_checkpoint(output_dir, prefix):
    """Find the latest checkpoint file matching prefix in output_dir.

    Args:
        output_dir: Directory to search
        prefix: Checkpoint filename prefix (e.g., "scale_batch_checkpoint_step")

    Returns:
        (path, step) tuple, or (None, 0) if no checkpoint found
    """
    pattern = os.path.join(output_dir, f"{prefix}*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0

    def get_step(path):
        fname = os.path.basename(path)
        step_str = fname.replace(prefix, "").replace(".pt", "")
        return int(step_str)

    latest = max(checkpoints, key=get_step)
    return latest, get_step(latest)


def load_checkpoint(path, model, config):
    """Load checkpoint and verify config matches.

    IMPORTANT: Call this BEFORE torch.compile(model)!

    Args:
        path: Path to checkpoint file
        model: Model to load weights into (not yet compiled)
        config: Expected config dict to verify against

    Returns:
        checkpoint_data dict (contains optimizer_state_dict, step, etc.)

    Raises:
        ValueError: If config doesn't match
    """
    checkpoint_data = torch.load(path, map_location='cpu', weights_only=False)

    # Verify config matches
    saved_config = checkpoint_data.get('config', {})
    if saved_config:
        for key, value in config.items():
            saved_value = saved_config.get(key)
            if saved_value != value:
                raise ValueError(
                    f"Config mismatch: {key}={saved_value} (checkpoint) vs {value} (current)"
                )

    model.load_state_dict(checkpoint_data['model_state_dict'])
    return checkpoint_data


def restore_optimizer(optimizer, checkpoint_data, device):
    """Restore optimizer state from checkpoint and move to device.

    Args:
        optimizer: Optimizer (e.g., SAM wrapper)
        checkpoint_data: Dict from load_checkpoint
        device: Target device (e.g., 'cuda')
    """
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    # Move optimizer state tensors to device
    # Handle SAM's nested base_optimizer
    base_opt = getattr(optimizer, 'base_optimizer', optimizer)
    for state in base_opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_checkpoint(path, step, model, optimizer, config):
    """Save checkpoint with clean state dict (strips _orig_mod. prefix).

    Args:
        path: Output path
        step: Current training step
        model: Model (may be compiled)
        optimizer: Optimizer
        config: Config dict to save for verification on resume
    """
    # Strip _orig_mod. prefix from compiled model keys
    state_dict = {
        k.replace('_orig_mod.', ''): v
        for k, v in model.state_dict().items()
    }

    checkpoint = {
        'step': step,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    torch.save(checkpoint, path)
