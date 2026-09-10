# TAIT — Termux AI Training

**A lightweight, NumPy-first local AI training toolkit for Termux, Android, and constrained Linux devices.**  


[English](README.md) · [العربية](README.ar.md) · [简体中文](README.zh-CN.md)

TAIT started as a single-file MiniGPT trainer and is now organized as an open-source Python package while keeping the same lightweight philosophy: no PyTorch, no TensorFlow, and no large runtime stack.

## ✨ Features

- NumPy-only MiniGPT training and inference
- Byte-level BPE tokenizer
- JSONL datasets with common prompt/response aliases
- Optional Hugging Face dataset adapter
- TOML configuration with CLI overrides
- First-run environment wizard with Termux-aware NumPy installation
- `doctor` and `benchmark` diagnostics
- Terminal chat and local browser chat (`--web`)
- Checkpoints and `.npz` model files
- Contributor-friendly modular source tree

## 🚀 Installation

### Python / pip

```bash
pip install tait
```

### From source

```bash
git clone <YOUR_REPOSITORY_URL>
cd TAIT
pip install -e .
```

### Termux note

Termux installations should prefer the Termux package for NumPy when available:

```bash
pkg install python python-numpy
```

Then run:

```bash
tait setup
```

The setup wizard runs automatically on the first normal TAIT execution and records completion. Re-run it explicitly with `tait setup --repair`.

## ⚡ Quick Start

```bash
tait doctor
tait benchmark
```

Train a tiny model:

```bash
tait train --data examples/datasets/demo.jsonl --epochs 3 --output demo.npz
```

Chat in the terminal:

```bash
tait chat --model demo.npz
```

Open the local browser UI:

```bash
tait chat --web --model demo.npz
```

## 🧠 Training

Configuration files are TOML and command-line options override config values:

```bash
tait train --config configs/tiny.toml
```

See [Training](docs/training.md) and [Configuration](docs/configuration.md) for the full reference.

## 📦 Datasets

Each JSONL line can use `prompt` / `response`, or the compatible aliases `question` / `answer` and `input` / `output`.

```json
{"prompt":"Hello","response":"Hi!"}
```

Inspect or validate a dataset:

```bash
tait data inspect examples/datasets/demo.jsonl
tait data validate examples/datasets/demo.jsonl --strict
```

See [Datasets](docs/datasets.md) for details and the optional Hugging Face adapter.

## 🌐 Web Chat

`--web` starts a local HTTP server on `127.0.0.1` and uses the browser for Unicode-friendly chat, including Arabic, English, and Chinese. The UI uses only HTML/CSS/vanilla JavaScript.

See [Web Chat](docs/web-chat.md).

## 🤝 Contributing

The project is intentionally modular so contributors can work on the CLI, model engine, datasets, configuration, hardware integration, or web UI independently.

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [Development](docs/development.md) before opening a pull request.

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

TAIT v2 focuses on a clean lightweight training/chat foundation. A future v3 can add an **Agent** layer, function calling, tools, planning, and execution without forcing those concerns into the v2 core.

See [Roadmap](docs/roadmap.md).

## ❤️ Support

Support and donation details will be published here once an official method is selected.

**Made in Sudan 🇸🇩**

## License

MIT — see [LICENSE](LICENSE).
