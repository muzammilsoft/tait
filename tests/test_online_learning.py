"""Tests for online / inference-time learning: MiniGPT.learn().

Covers: a learning step executes, weights really change, loss on the
example goes down, generation behavior changes, save/reload preserves the
learning, and the traditional training + inference paths still work.
"""
import numpy as np

from tait.core.model import MiniGPT
from tait.core.tokenizer import BPETokenizer
from tait.core.generation import generate_reply


def _tiny_trained_pair(seed=7):
    """A small model + tokenizer, briefly trained on two toy examples."""
    tok = BPETokenizer()
    tok.train(
        ["alpha beta gamma delta", "one two three four",
         "the sky is blue today", "cats drink milk daily"],
        vocab_size=280, sample_size=40, show_progress=False)
    model = MiniGPT(tok.vocab_size_total, tok.pad_id,
                    d_model=8, d_ff=16, block_size=24, seed=seed)
    return model, tok


def _gen_kwargs():
    # Near-greedy + fixed seed => deterministic generation for tests.
    return dict(max_new_tokens=12, temperature=0.01, top_k=4, top_p=1.0, seed=123)


def test_learn_executes_and_reports_losses():
    model, tok = _tiny_trained_pair()
    out = model.learn(tok, "The capital of Numa is", "Zorak",
                      steps=5, lr=3e-3)
    assert set(out) == {"loss_before", "loss_after", "losses", "steps",
                        "lr", "target_tokens"}
    assert out["steps"] == 5
    assert len(out["losses"]) == 5
    assert all(np.isfinite(l) for l in out["losses"])
    assert out["target_tokens"] > 0


def test_learn_changes_weights():
    model, tok = _tiny_trained_pair()
    before = {p: getattr(model, p).copy() for p in model.params}
    model.learn(tok, "The capital of Numa is", "Zorak", steps=5, lr=3e-3)
    changed = [p for p in model.params
               if not np.array_equal(before[p], getattr(model, p))]
    assert changed, "learn() must modify at least one parameter array"


def test_learn_reduces_loss_on_example():
    model, tok = _tiny_trained_pair()
    out = model.learn(tok, "The capital of Numa is", "Zorak",
                      steps=20, lr=3e-3)
    assert out["loss_after"] < out["loss_before"], (
        f"loss did not decrease: {out['loss_before']:.4f} -> "
        f"{out['loss_after']:.4f}")


def test_learn_changes_generation_output():
    model, tok = _tiny_trained_pair()
    prompt = "The capital of Numa is"
    before = generate_reply(model, tok, prompt, **_gen_kwargs())
    model.learn(tok, prompt, "Zorak", steps=100, lr=3e-3)
    after = generate_reply(model, tok, prompt, **_gen_kwargs())
    assert after != before, "generation output should change after learning"
    assert "Zorak" in after, (
        f"expected the learned fact in the reply, got: {after!r}")


def test_learn_zero_steps_changes_nothing():
    model, tok = _tiny_trained_pair()
    before = {p: getattr(model, p).copy() for p in model.params}
    out = model.learn(tok, "The capital of Numa is", "Zorak", steps=0)
    assert out["loss_before"] == out["loss_after"]
    for p in model.params:
        assert np.array_equal(before[p], getattr(model, p))


def test_save_reload_preserves_learning(tmp_path):
    model, tok = _tiny_trained_pair()
    prompt = "The capital of Numa is"
    model.learn(tok, prompt, "Zorak", steps=100, lr=3e-3)
    path = tmp_path / "learned.npz"
    model.save(path, tok)
    loaded, loaded_tok = MiniGPT.load(path)
    # The learned behavior survives the round-trip.
    reply = generate_reply(loaded, loaded_tok, prompt, **_gen_kwargs())
    assert "Zorak" in reply, f"reload lost the learning, got: {reply!r}"
    # And the loss on the learned example is still low after reload.
    probe = loaded.learn(loaded_tok, prompt, "Zorak", steps=0)
    assert probe["loss_before"] < 1.0, probe["loss_before"]


def test_traditional_training_still_works(tmp_path):
    """End-to-end training via the real cmd_train path on a tiny dataset."""
    import json
    from argparse import Namespace
    from tait.cli.commands import cmd_train

    data = tmp_path / "tiny.jsonl"
    rows = [
        {"prompt": "sky color?", "response": "The sky is blue."},
        {"prompt": "grass color?", "response": "The grass is green."},
        {"prompt": "sun color?", "response": "The sun is yellow."},
        {"prompt": "rose color?", "response": "The rose is red."},
    ]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "trained.npz"
    args = Namespace(
        data=str(data), no_wake_lock=True, validation_split=0.0, seed=42,
        vocab_size=280, bpe_sample_size=40, reasoning=False,
        cache_dir=str(tmp_path / "cache"), max_seq_len=None,
        batch_size=2, response_only_loss=True, d_model=8, d_ff=16,
        epochs=3, lr=3e-3, checkpoint_every=0, checkpoint_path=str(tmp_path / "ckpt.npz"),
        early_stopping=False, min_delta=0.001,
        best_checkpoint_path=str(tmp_path / "best.npz"), patience=3,
        output=str(out))
    cmd_train(args)
    assert out.exists()
    model, tok = MiniGPT.load(str(out))
    reply = generate_reply(model, tok, "sky color?",
                           max_new_tokens=10, temperature=0.01,
                           top_k=4, top_p=1.0, seed=1)
    assert isinstance(reply, str) and len(reply) > 0


def test_inference_unaffected_without_learning():
    model, tok = _tiny_trained_pair()
    r1 = generate_reply(model, tok, "the sky is", **_gen_kwargs())
    r2 = generate_reply(model, tok, "the sky is", **_gen_kwargs())
    assert r1 == r2, "inference must be deterministic given a fixed seed"
    assert isinstance(r1, str)
