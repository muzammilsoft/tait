import numpy as np
from tait.core.model import MiniGPT
from tait.core.tokenizer import BPETokenizer


def test_model_forward_backward_and_roundtrip(tmp_path):
    tok = BPETokenizer(); tok.train(["a b", "a c", "hello"], vocab_size=280, sample_size=20, show_progress=False)
    model = MiniGPT(tok.vocab_size_total, tok.pad_id, d_model=8, d_ff=16, block_size=8, seed=1)
    ids = np.array([tok.encode("a")[:3]], dtype=np.int64)
    mask = np.ones_like(ids, dtype=bool)
    probs, cache = model.forward(ids, mask)
    grads = model.backward(cache, ids, mask)
    assert probs.shape[:2] == ids.shape
    model.step(grads, lr=1e-3)
    path = tmp_path / "m.npz"
    model.save(path, tok)
    loaded, loaded_tok = MiniGPT.load(path)
    assert loaded.wte.shape == model.wte.shape
    assert loaded_tok.decode(loaded_tok.encode("hello")) == "hello"
