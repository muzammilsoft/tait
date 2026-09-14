# Configuration

TAIT uses TOML because it is available in modern Python through the standard library, keeping the core dependency footprint small.

Example:

```toml
[train]
data = "examples/datasets/demo.jsonl"
epochs = 3
batch_size = 4
d_model = 16
d_ff = 32
```

Use it with:

```bash
tait train --config configs/tiny.toml
```

Precedence is: **CLI options > config file > built-in defaults**.


## Reasoning and validation

The standard v2.1 defaults enable compact reasoning, response-only loss, a 10% validation split, and early stopping. Useful controls include:

```toml
reasoning = true
response_only_loss = true
validation_split = 0.1
early_stopping = true
patience = 10
min_delta = 0.001
best_checkpoint_path = "checkpoint_best.npz"
seed = 42
```

Set `reasoning = false` for direct prompt/response training. Set `response_only_loss = false` if you intentionally want to train on prompt tokens as well.
