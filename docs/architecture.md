# Architecture

TAIT v2 separates the user-facing CLI from the NumPy model engine, dataset sources, configuration, hardware helpers, setup, and web UI.

```text
CLI
├── setup / doctor / benchmark
├── train / chat / data
│
├── Config
├── DataSource
│   ├── Local JSONL
│   └── Optional Hugging Face
├── Core
│   ├── BPE tokenizer
│   ├── MiniGPT
│   ├── generation
│   └── checkpoints
├── Hardware
│   └── Termux helpers
└── Web
    └── stdlib HTTP server + vanilla JS
```

The v2 core intentionally has no Agent/function-calling layer. That boundary leaves room for a v3 Agent subsystem without forcing tools or execution policy into the training engine.
