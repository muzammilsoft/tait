# Models

The current model is a small single-head MiniGPT implemented directly with NumPy. It contains token/position embeddings, causal self-attention, a feed-forward block, residual connections, and an output head.

Models are stored as `.npz` archives containing weights plus BPE merge information and tokenizer special-token IDs. This keeps inference independent of the training dataset.

Future model families should expose clean load/generate boundaries so the CLI does not need to know model internals.
