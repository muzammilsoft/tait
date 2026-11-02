# Changelog

## 2.3.0

- Added Hugging Face publishing: `tait hf login` (token typed hidden, stored with mode 0600, never printed or logged), `tait hf logout`, `tait hf status`, `tait hf config`, `tait publish --model <file.npz> --repo user/model`, and `tait train --publish --publish-repo user/model`.
- Training now writes a `<model>.meta.json` sidecar with real metadata (TAIT version, dataset name only, examples, epochs, final and validation loss, shapes, seed, date) so the model card is generated from facts, not invented claims.
- Publishing generates an honest model card (README.md): TAIT `.npz` format, shapes, file size and hash, training data and metrics when known, how to run it with `tait chat` or the KAIT runner, and a clear limitations note that tiny models may hallucinate.
- The `.npz` model format is unchanged: old TAIT models publish without retraining, and KAIT reads the same file.
- Core training and chat still need only NumPy. Publishing uses the official `huggingface_hub` client only, via the optional extra `pip install "tait[publish]"`.
- Fixed the version drift where `tait version` reported 2.1.0 while the package was 2.2.0; both now report 2.3.0.

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
