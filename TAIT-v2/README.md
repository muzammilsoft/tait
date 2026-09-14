# TAIT — Termux AI Training

**A lightweight, NumPy-first local AI training toolkit for Termux, Android, and constrained Linux devices.**  
**Made in Sudan 🇸🇩**

[English](README.md) · [العربية](README.ar.md) · [简体中文](README.zh-CN.md)

TAIT started as a single-file MiniGPT trainer and is now an open-source modular package. It stays deliberately lightweight: no PyTorch, no TensorFlow, and no large runtime stack.

## ✨ Features

- NumPy-only MiniGPT training and inference
- Byte-level BPE tokenizer
- Compact reasoning / CoT training format
- Response-only loss and validation loss
- Early stopping and best checkpoints
- Safe Ctrl+C checkpoint preservation
- JSONL datasets with reasoning aliases
- Arabic, English, and mixed sample datasets
- Optional Hugging Face dataset adapter
- TOML configuration with CLI overrides
- Termux-aware setup, doctor, and benchmark
- Terminal chat and local browser chat

## 🚀 Installation

```bash
pip install tait
```

From source:

```bash
git clone <YOUR_REPOSITORY_URL>
cd TAIT
pip install -e .
```

## ⚡ Quick Start

From the repository:

```bash
tait data inspect examples/datasets/mixed.jsonl
tait train --config configs/reasoning.toml
tait chat --model artifacts/reasoning.npz
```

Browser chat:

```bash
tait chat --web --model artifacts/reasoning.npz
```

Show the generated compact reasoning trace:

```bash
tait chat --model artifacts/reasoning.npz --show-reasoning
```

## 📦 Dataset format

```json
{"prompt":"شنو عاصمة السودان؟","reasoning":"السؤال عن عاصمة السودان، والمدينة المعروفة بأنها العاصمة هي الخرطوم.","response":"عاصمة السودان هي الخرطوم."}
```

The default dataset is `examples/datasets/mixed.jsonl`. Arabic and English datasets are included separately. See [Datasets](docs/datasets.md) for detailed generation rules.

## 🧠 Training

```bash
tait train --config configs/default.toml
```

CLI options override TOML settings. Validation, early stopping, response-only loss, and compact reasoning are enabled by default in the standard configuration.

See [Training](docs/training.md).

## 📚 Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [Models](docs/models.md)
- [Web Chat](docs/web-chat.md)
- [Termux](docs/termux.md)
- [Development](docs/development.md)
- [Coding Guidelines](docs/coding-guidelines.md)
- [AI Contributions](docs/ai-contributions.md)
- [Release Process](docs/release-process.md)
- [Roadmap](docs/roadmap.md)

## 🗺️ Roadmap

v2.1 focuses on better instruction learning and compact reasoning. A future v2.x can add teacher/student distillation workflows. v3 can add an **Agent** layer with function calling, tools, planning, and execution.

## License

MIT — see [LICENSE](LICENSE).
