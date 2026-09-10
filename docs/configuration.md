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
