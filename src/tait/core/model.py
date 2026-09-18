"""NumPy-first MiniGPT model. Derived from TAIT's original single-file engine."""
import os
import numpy as np
from .tokenizer import BPETokenizer, SEP, END

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def _single_example_batch(ids, pad_id, block_size, sep_id, response_only=True):
    """Build the (x, y, valid, tgt_valid) batch for one learn() example.

    Same layout as the training batches: x = s[:-1], y = s[1:], with the
    loss mask covering only the response tokens after SEP when
    response_only is True. Sequences longer than block_size + 1 are
    truncated, exactly like training does.
    """
    s = ids[:block_size + 1]
    L = len(s) - 1
    x = np.full((1, L), pad_id, dtype=np.int64)
    y = np.full((1, L), pad_id, dtype=np.int64)
    valid = np.zeros((1, L), dtype=bool)
    tgt_valid = np.zeros((1, L), dtype=bool)
    x[0, :L], y[0, :L] = s[:-1], s[1:]
    valid[0, :L] = True
    if response_only and sep_id in s:
        tgt_valid[0, s.index(sep_id):L] = True
    else:
        tgt_valid[0, :L] = True
    return x, y, valid, tgt_valid


class MiniGPT:
    def __init__(self, vocab_size, pad_id, d_model=32, d_ff=64, block_size=64, seed=42):
        rng = np.random.default_rng(seed)
        self.V, self.D, self.Dff, self.block_size, self.pad_id = vocab_size, d_model, d_ff, block_size, pad_id

        def xavier(fan_in, fan_out):
            return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)

        self.wte = xavier(vocab_size, d_model)
        self.wpe = xavier(block_size, d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = (xavier(d_model, d_model) for _ in range(4))
        self.W1 = xavier(d_model, d_ff)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = xavier(d_ff, d_model)
        self.b2 = np.zeros(d_model, dtype=np.float32)
        self.Whead = xavier(d_model, vocab_size)
        self.bhead = np.zeros(vocab_size, dtype=np.float32)

        self.params = ["wte", "wpe", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "Whead", "bhead"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, x_batch, valid_mask):
        B, T = x_batch.shape
        D = self.D
        tok_emb = self.wte[x_batch]
        pos_emb = self.wpe[:T]
        h0 = tok_emb + pos_emb

        Q = h0 @ self.Wq
        K = h0 @ self.Wk
        Vv = h0 @ self.Wv

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D)
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        key_pad = ~valid_mask[:, None, :]
        mask = causal[None, :, :] | key_pad
        scores = np.where(mask, -1e9, scores)
        A = softmax(scores, axis=-1)

        attn_out = A @ Vv
        attn_proj = attn_out @ self.Wo
        h1 = h0 + attn_proj

        pre1 = h1 @ self.W1 + self.b1
        ff1 = np.maximum(0, pre1)
        ff2 = ff1 @ self.W2 + self.b2
        h2 = h1 + ff2

        logits = h2 @ self.Whead + self.bhead
        probs = softmax(logits, axis=-1)

        cache = dict(x_batch=x_batch, valid_mask=valid_mask, h0=h0, Q=Q, K=K, Vv=Vv,
                     A=A, attn_out=attn_out, h1=h1, pre1=pre1, ff1=ff1, h2=h2, probs=probs, T=T, B=B)
        return probs, cache

    def backward(self, cache, targets, target_mask):
        B, T, D = cache["B"], cache["T"], self.D
        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        n_valid = max(target_mask.sum(), 1)

        probs = cache["probs"].copy()
        onehot_idx = np.clip(targets, 0, self.V - 1)
        probs[np.arange(B)[:, None], np.arange(T)[None, :], onehot_idx] -= 1.0
        dlogits = probs * target_mask[:, :, None] / n_valid

        grads["Whead"] = np.einsum("btd,btv->dv", cache["h2"], dlogits)
        grads["bhead"] = dlogits.sum(axis=(0, 1))
        dh2 = dlogits @ self.Whead.T

        d_ff2 = dh2
        grads["W2"] = np.einsum("btf,btd->fd", cache["ff1"], d_ff2)
        grads["b2"] = d_ff2.sum(axis=(0, 1))
        d_ff1 = d_ff2 @ self.W2.T
        d_pre1 = d_ff1 * (cache["pre1"] > 0)
        grads["W1"] = np.einsum("btd,btf->df", cache["h1"], d_pre1)
        grads["b1"] = d_pre1.sum(axis=(0, 1))
        d_h1_from_ffn = d_pre1 @ self.W1.T

        d_h1 = dh2 + d_h1_from_ffn

        d_attn_proj = d_h1
        grads["Wo"] = np.einsum("btd,bte->de", cache["attn_out"], d_attn_proj)
        d_attn_out = d_attn_proj @ self.Wo.T

        A = cache["A"]
        dA = d_attn_out @ cache["Vv"].transpose(0, 2, 1)
        dVv = A.transpose(0, 2, 1) @ d_attn_out

        dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(D)

        dQ = dscores @ cache["K"]
        dK = dscores.transpose(0, 2, 1) @ cache["Q"]

        grads["Wq"] = np.einsum("btd,bte->de", cache["h0"], dQ)
        grads["Wk"] = np.einsum("btd,bte->de", cache["h0"], dK)
        grads["Wv"] = np.einsum("btd,bte->de", cache["h0"], dVv)

        d_h0_from_attn = dQ @ self.Wq.T + dK @ self.Wk.T + dVv @ self.Wv.T
        d_h0 = (d_h1 + d_h0_from_attn) * cache["valid_mask"][:, :, None]

        x_flat = cache["x_batch"].reshape(-1)
        d_h0_flat = d_h0.reshape(-1, D)
        np.add.at(grads["wte"], x_flat, d_h0_flat)
        grads["wpe"][:T] += d_h0.sum(axis=0)
        return grads

    def step(self, grads, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g * g)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def learn(self, tokenizer, prompt, response, reasoning="",
              enable_reasoning=False, steps=10, lr=3e-3,
              response_only=True, verbose=False):
        """Online / inference-time learning on a single example.

        Encodes ``prompt`` + ``response`` exactly like training does
        (``format_example`` serialization, SEP/END framing, response-only
        loss mask), then runs ``steps`` forward/backward/Adam steps on that
        one example using the model's own optimizer state.

        This mutates the model's weights in place. It is memorization-scale
        learning, not a replacement for training: there is no validation
        split, no early stopping, and nothing is saved automatically -- call
        ``save()`` yourself if you want to keep the updated weights.

        Online learning != AGI, != guaranteed reasoning, != automatic
        self-improvement. It is a research hook for studying continual
        weight updates at inference time.

        Returns a dict with ``loss_before``, ``loss_after``, ``losses``
        (per-step), ``steps``, ``lr`` and ``target_tokens``.
        """
        from .reasoning import format_example

        prompt_text, target_text = format_example(
            prompt, response, reasoning, enable_reasoning)
        ids = tokenizer.encode(prompt_text + SEP + target_text + END)
        x, y, valid, tgt_valid = _single_example_batch(
            ids, self.pad_id, self.block_size, tokenizer.sep_id,
            response_only=response_only)

        def _nll(probs):
            p = np.clip(probs, 1e-9, 1.0)
            yi = np.clip(y, 0, self.V - 1)
            ll = np.log(np.take_along_axis(p, yi[..., None], axis=-1)[..., 0])
            return float(-(ll * tgt_valid).sum() / max(tgt_valid.sum(), 1))

        probs, _ = self.forward(x, valid)
        loss_before = _nll(probs)
        losses = []
        for _ in range(max(int(steps), 0)):
            probs, cache = self.forward(x, valid)
            loss = _nll(probs)
            losses.append(loss)
            grads = self.backward(cache, y, tgt_valid)
            self.step(grads, lr=lr)
        probs, _ = self.forward(x, valid)
        loss_after = _nll(probs)
        result = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "losses": losses,
            "steps": len(losses),
            "lr": lr,
            "target_tokens": int(tgt_valid.sum()),
        }
        if verbose:
            print("learn: loss_before=%.4f loss_after=%.4f steps=%d lr=%g "
                  "target_tokens=%d" % (
                      loss_before, loss_after, len(losses), lr,
                      result["target_tokens"]))
        return result

    def forward_single(self, ids_cond):
        x = np.array([ids_cond])
        mask = np.ones_like(x, dtype=bool)
        probs, _ = self.forward(x, mask)
        return probs[0]

    def generate(self, tokenizer, prompt, max_new_tokens=60, temperature=0.7, top_k=8, top_p=0.9, seed=None):
        rng = np.random.default_rng(seed)
        ids = tokenizer.encode(prompt + SEP)
        for _ in range(max_new_tokens):
            ids_cond = ids[-self.block_size:]
            probs = self.forward_single(ids_cond)[-1].copy()
            probs[self.pad_id] = 0.0
            probs /= probs.sum()

            p = probs ** (1.0 / max(temperature, 1e-6))
            p /= p.sum()
            if top_k and top_k < len(p):
                keep = np.argpartition(p, -top_k)[-top_k:]
                m = np.zeros_like(p, dtype=bool); m[keep] = True
                p = np.where(m, p, 0.0)
            if top_p:
                order = np.argsort(-p)
                cum = np.cumsum(p[order])
                cutoff = np.searchsorted(cum, top_p) + 1
                keep = order[:cutoff]
                m = np.zeros_like(p, dtype=bool); m[keep] = True
                p = np.where(m, p, 0.0)
            p /= p.sum()
            next_id = int(rng.choice(len(p), p=p))
            ids.append(next_id)
            if next_id == tokenizer.end_id:
                break
        return tokenizer.decode(ids)

    def save(self, path, tokenizer):
        data = {p: getattr(self, p) for p in self.params}
        pairs = sorted(tokenizer.merges.items(), key=lambda kv: kv[1])
        data["merge_a"] = np.array([a for (a, b), _ in pairs], dtype=np.int32)
        data["merge_b"] = np.array([b for (a, b), _ in pairs], dtype=np.int32)
        data["merge_id"] = np.array([nid for _, nid in pairs], dtype=np.int32)
        data["sep_id"] = tokenizer.sep_id
        data["end_id"] = tokenizer.end_id
        data["tok_pad_id"] = tokenizer.pad_id
        data["block_size"] = self.block_size
        data["pad_id"] = self.pad_id
        np.savez(path, **data)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise SystemExit(f"Model file not found: {path}")
        data = np.load(path, allow_pickle=True)
        if "merge_a" not in data.files:
            raise SystemExit(f"'{path}' is not a valid TAIT model file (missing BPE merge data).")
        tok = BPETokenizer()
        for a, b, nid in zip(data["merge_a"], data["merge_b"], data["merge_id"]):
            a, b, nid = int(a), int(b), int(nid)
            tok.merges[(a, b)] = nid
            tok.vocab[nid] = tok.vocab[a] + tok.vocab[b]
        tok.sep_id, tok.end_id, tok.pad_id = int(data["sep_id"]), int(data["end_id"]), int(data["tok_pad_id"])
        d_model = data["wte"].shape[1]
        d_ff = data["W1"].shape[1]
        block_size = int(data["block_size"])
        model = MiniGPT(tok.vocab_size_total, pad_id=int(data["pad_id"]), d_model=d_model,
                         d_ff=d_ff, block_size=block_size)
        for p in model.params:
            setattr(model, p, data[p])
        return model, tok
