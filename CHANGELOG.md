# Changelog

## 2.2.0

- Added online/inference-time learning via `MiniGPT.learn(prompt, response, steps, lr)`.
- Added 8 tests covering learning behavior, weight updates, save/reload persistence, and training/inference compatibility.
- Added online learning documentation (`docs/online-learning.md`) with honest experiment results and limitations.
- Added reproducible online learning experiment (`experiments/online_learning_poc.py`) and learning-rate sweep (`experiments/learn_sweep.py`).
- Added sample datasets for online learning experiments.
- Added GitHub Actions workflow for automated PyPI publishing on release.

## 2.1.0

- Added optional compact reasoning/CoT dataset support.
- Added response-only loss masking.
- Added validation split and validation loss reporting.
- Added early stopping and best-checkpoint restoration.
- Added safe Ctrl+C checkpoint preservation.
- Added Arabic, English, and mixed reasoning dataset samples.
- Updated default training configuration to the mixed sample dataset.
- Added reasoning-aware chat parsing and `--show-reasoning`.
- Added training and dataset documentation for high-quality instruction data.

## 2.0.0

- Modular NumPy-first package architecture.
- CLI, TOML configuration, datasets, checkpoints, Termux support, and local web chat.
