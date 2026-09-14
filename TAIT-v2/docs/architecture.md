# Architecture

TAIT keeps the model core deliberately small:

```text
CLI
 ├── setup / doctor / benchmark
 ├── train
 │    ├── JSONL / data sources
 │    ├── BPE tokenizer
 │    ├── instruction serializer
 │    ├── response-only loss
 │    ├── validation evaluator
 │    └── checkpoint manager
 └── chat
      ├── terminal
      └── local web UI

Core
 ├── MiniGPT
 ├── BPETokenizer
 ├── generation
 └── reasoning serializer/parser
```

Reasoning is a training/generation capability, not an Agent system. Function calling, tools, planning, and execution remain future v3 concerns.
