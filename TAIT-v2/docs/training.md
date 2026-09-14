# Training

TAIT 2.1 trains the current from-scratch MiniGPT engine using a byte-level BPE tokenizer and NumPy operations.

Typical run:

```bash
tait train --config configs/reasoning.toml
```

Or:

```bash
tait train --data examples/datasets/mixed.jsonl --epochs 10
```

## Reasoning / CoT mode

Reasoning is enabled by default in the v2.1 configuration. A dataset row with `reasoning` becomes a compact `<think>...</think>` section followed by `<answer>...</answer>`.

This is deliberately a **compact reasoning trace**, not a requirement for long hidden deliberations. Tiny models have limited capacity, so concise task-relevant reasoning is preferred.

## Validation and early stopping

By default TAIT reserves 10% of the examples for validation, reports both losses, saves the best validation checkpoint, and stops after 10 epochs without sufficient improvement.

```toml
validation_split = 0.1
patience = 10
min_delta = 0.001
early_stopping = true
```

## Response-only loss

`response_only_loss = true` is enabled by default. The model still sees the prompt, but training loss starts after `SEP`, focusing optimization on generating the target.

## Safe abort

Pressing `Ctrl+C` during model training now saves the current model to `checkpoint_path` before exiting with code 130. The most recent completed periodic checkpoint is also preserved when enabled.

## Scaling on phones

Start small. Increase `d_model`, `d_ff`, vocabulary, or sequence length one at a time and benchmark. More epochs are not automatically better; use validation loss to detect overfitting.
