import math
import time

from ..core.model import MiniGPT
from ..core.tokenizer import BPETokenizer


def run_benchmark(d_model=32, d_ff=64, batch_size=64, seq_len=64, dataset_size=5000, epochs=60):
    import numpy as np
    tok = BPETokenizer(); tok.sep_id, tok.end_id, tok.pad_id = 256, 257, 258
    model = MiniGPT(vocab_size=300, pad_id=tok.pad_id, d_model=d_model, d_ff=d_ff, block_size=seq_len)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 250, size=(batch_size, seq_len))
    y = rng.integers(0, 250, size=(batch_size, seq_len))
    valid = np.ones_like(x, dtype=bool)
    model.backward(model.forward(x, valid)[1], y, valid)
    n_iters = 5
    t0 = time.time()
    for _ in range(n_iters):
        probs, cache = model.forward(x, valid)
        grads = model.backward(cache, y, valid)
        model.step(grads)
    dt = (time.time() - t0) / n_iters
    n_batches = math.ceil(dataset_size / batch_size)
    epoch_est = dt * n_batches
    print(f"Average time per batch: {dt*1000:.2f} ms")
    print(f"Estimated batches/epoch: {n_batches}")
    print(f"Estimated time/epoch: {epoch_est:.1f}s (~{epoch_est/60:.1f} min)")
    print(f"Estimated time for {epochs} epochs: ~{epoch_est*epochs/60:.1f} min")
