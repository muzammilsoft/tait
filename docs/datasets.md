# Datasets

## JSONL

Each non-empty line must be a JSON object. TAIT accepts these field aliases:

- prompt: `prompt`, `question`, or `input`
- response: `response`, `answer`, or `output`

Invalid JSON and rows missing either side are skipped by the loader.

Useful commands:

```bash
tait data inspect dataset.jsonl
tait data validate dataset.jsonl --strict
```

## Hugging Face

The adapter is optional so the base installation stays small:

```bash
pip install "tait[hf]"
```

The adapter is deliberately isolated from the trainer through the `DataSource` interface.
