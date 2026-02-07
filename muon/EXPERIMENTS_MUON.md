# Muon Experiments History

This document tracks Muon usage in this repo, from the original Kaggle-era scripts through the recent reintroduction attempts on sudoku‑extreme.

## Early Muon usage (Kaggle / pre‑sudoku‑extreme pipeline)

Muon was originally added in the early training scripts and ablation experiments, typically with a **split optimizer**:
- **Muon for ≥2D parameters**
- **AdamW for 1D parameters** (biases, norms, embeddings)

Notable commits that include Muon usage (from `git log -S "optimizer_muon"`):
- `f147270` — *Muon made loss go down _hard_. But still saturates at the same spot (not its fault)*
- `86439e7` — *Add ablation study scripts* (Muon still present)
- `7c13903` — *Add experiment documentation and additional transformer experiments*
- `9f81177` — *Add projection experiments, debunk project‑then‑add results*
- `d27f724` — *Add sinusoidal pos encoding experiment - complete failure*
- `d13b5bb` — *Add batch size scaling experiments (BS=256 best)*

These are all from the original Kaggle/easy‑only era (before sudoku‑extreme and curriculum runs).

## Shift away from Muon (sudoku‑extreme pipeline)

The curriculum training pipeline (sudoku‑extreme) landed in:
- `670df77` — *Add curriculum learning experiment: reverse (hard→easy) wins*

From that point onward, new experiments were based on **SAM/AdamW** rather than Muon. Muon was not explicitly removed, but it stopped being used in the newer experiment family.

## Muon sweep summary (sudoku‑extreme, 2026‑02‑05+)

All runs: d_model=128, n_layers=4, n_iterations=16, batch_size=4096, cosine LR. Muon weight_decay=0.0 unless noted.

- Reverse curriculum (hard→easy), all‑Muon, no grad clip: lr 1.5e‑3 (75.8%), 2e‑3 (78.8%), 3e‑3 (80.8%), 5e‑3 (82.9%), 7e‑3 (82.9%), 9e‑3 (running).
- Reverse curriculum, all‑Muon, grad clip=1.0: lr 2e‑2 (exploded).
- Mixed sampling (no phases), all‑Muon, no grad clip: lr 1e‑2 and 2e‑2 (exploded).
- Mixed sampling, all‑Muon, grad clip=1.0: lr 1e‑2 (exploded).
- Mixed sampling, split Muon>=2D + AdamW 1D, no grad clip: muon lr 1e‑2, adamw lr 1.5e‑3 (exploded).

## Current choice

Reverse curriculum, all‑Muon, lr=5e‑3, no grad clip, no mixed, no split. lr=7e‑3 matched it; lr=9e‑3 still running.

## Conclusion

Muon matches the AdamW baseline at best and adds instability/complexity. We’re not adopting it as the default optimizer.

## Notes on log locations

- Modal logs are stored in the `sudoku-outputs` volume (download with `modal volume get`).
- Local Muon logs are not tracked in‑repo yet; download as needed.
