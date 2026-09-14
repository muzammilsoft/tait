# TAIT — Termux AI Training

**一个轻量、以 NumPy 为核心的本地 AI 训练工具，面向 Termux、Android 和资源受限设备。**  
**Made in Sudan 🇸🇩**

[English](README.md) · [العربية](README.ar.md) · [简体中文](README.zh-CN.md)

TAIT 从一个单文件 MiniGPT 训练脚本发展成了模块化的 Python 开源项目，同时保持轻量理念：不依赖 PyTorch、TensorFlow 或大型运行时。

## ✨ 功能

- NumPy MiniGPT 训练与推理
- Byte-level BPE tokenizer
- JSONL 数据集
- 可选 Hugging Face 数据源
- TOML 配置与 CLI 覆盖
- 首次运行设置向导，识别 Termux
- `doctor` 与 `benchmark`
- 终端聊天与本地 Web Chat
- `.npz` 模型与 checkpoint
- 适合开源贡献的模块化结构

## 🚀 快速安装

```bash
pip install tait
tait setup
```

Termux 推荐：

```bash
pkg install python python-numpy
```

## ⚡ 快速开始

```bash
tait doctor
tait benchmark
tait train --data examples/datasets/demo.jsonl --epochs 3 --output demo.npz
tait chat --model demo.npz
tait chat --web --model demo.npz
```

## 📚 文档

查看 [文档索引](docs/README.md) 以及 `docs/` 中的专题指南。

## 🤝 贡献

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 与[开发指南](docs/development.md)。

## 🗺️ 路线图

v2 专注于轻量、模块化的训练和聊天基础。未来 v3 可以加入 Agent、函数调用、工具、规划和执行，而不污染 v2 核心。

## ❤️ 支持

正式的赞助/捐赠方式确定后会在此公布。

## 许可证

MIT — 见 [LICENSE](LICENSE)。

## TAIT 2.1

2.1 版本加入了紧凑推理/CoT 数据格式、response-only loss、validation loss、early stopping，以及 Ctrl+C 安全保存 checkpoint，并提供阿拉伯语、英语和混合数据集示例。详见 `docs/datasets.md` 和 `docs/training.md`。
