# Training

TAIT trains the current from-scratch MiniGPT engine using a byte-level BPE tokenizer and NumPy operations.

Typical run:

```bash
tait train --data dataset.jsonl --epochs 10 --output model.npz
```

For constrained devices, start small (`d_model=16`, `d_ff=32`, modest batch/sequence lengths), benchmark, then scale carefully.

Checkpoints are written with `--checkpoint-every` and `--checkpoint-path`. BPE caches are stored under `--cache-dir`.
