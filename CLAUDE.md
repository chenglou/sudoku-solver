Don't forget to source venv before doing python commands

For training on Modal:

- Run with `--detach` so runs survive client disconnection
- Modal can preempt workers anytime (GPU jobs can't opt out). Auto-restarts with same input
- Implement checkpoint resumption: save model + optimizer + step + config, load before `torch.compile()`
- Save config in checkpoint to prevent loading wrong experiment's checkpoint
- `timeout` default is 5min. Training needs more. We can just use the max, 24h
- Volumes persist indefinitely (no auto-eviction)
- Local output stream can die while Modal worker continues. Check `modal container list` and `modal volume ls` for true status
- Keep training code Modal-agnostic: pass `output_dir` param, Modal wrapper sets it to volume path
