"""Proof-of-concept experiment: online / inference-time learning in TAIT.

Reproducible end-to-end run:
  1. Train a small base model on examples/datasets/online_learning_base.jsonl
  2. Probe BEFORE online learning (exact / paraphrase / control / forgetting)
  3. model.learn() on one new fictional fact
  4. Probe AFTER, save, reload, probe AFTER-RELOAD
  5. Write results to experiments/results/online_learning_poc.json

Generation settings are fixed for all probes so the comparison is fair.
"""
import json
import os
import platform
import sys
import time
from argparse import Namespace

import numpy as np

from tait.cli.commands import cmd_train
from tait.core.generation import generate_reply
from tait.core.model import MiniGPT

WORKDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WORKDIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed generation settings for every probe (fair before/after comparison).
GEN = dict(max_new_tokens=40, temperature=0.7, top_k=8, top_p=0.9, seed=42)

PROBES = [
    ("exact_recall", "What is the capital of the kingdom of Numa?"),
    ("paraphrase_1", "Which city is the capital of Numa?"),
    ("paraphrase_2", "Where is the capital of Numa?"),
    ("paraphrase_3", "Numa's capital is which city?"),
    ("control_1", "What color is grass?"),
    ("control_2", "What color is the sky?"),
    ("forget_A_1", "Which river flows from north to south?"),
    ("forget_A_2", "The Sila river flows in which direction?"),
]

LEARN_PROMPT = "What is the capital of the kingdom of Numa?"
LEARN_RESPONSE = "Zorak is the capital of the kingdom of Numa."
LEARN_STEPS = 100
LEARN_LR = 3e-3


def probe_all(model, tok):
    out = {}
    for name, q in PROBES:
        out[name] = {
            "question": q,
            "reply": generate_reply(model, tok, q, **GEN),
        }
    return out


def param_count(model):
    return int(sum(getattr(model, p).size for p in model.params))


def main():
    results = {
        "experiment": "online_learning_poc",
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "generation_settings": GEN,
        "learn": {"prompt": LEARN_PROMPT, "response": LEARN_RESPONSE,
                  "steps": LEARN_STEPS, "lr": LEARN_LR},
    }

    # 1. Train base model.
    base_path = os.path.join(WORKDIR, "base_model.npz")
    t0 = time.time()
    args = Namespace(
        data=os.path.join(WORKDIR, "..", "examples", "datasets",
                          "online_learning_base.jsonl"),
        no_wake_lock=True, validation_split=0.0, seed=42,
        vocab_size=300, bpe_sample_size=100, reasoning=False,
        cache_dir=os.path.join(WORKDIR, ".cache"), max_seq_len=None,
        batch_size=8, response_only_loss=True, d_model=32, d_ff=64,
        epochs=150, lr=3e-3, checkpoint_every=0,
        checkpoint_path=os.path.join(WORKDIR, "ckpt.npz"),
        early_stopping=False, min_delta=0.001,
        best_checkpoint_path=os.path.join(WORKDIR, "best.npz"),
        patience=10, output=base_path)
    cmd_train(args)
    train_time = time.time() - t0
    model, tok = MiniGPT.load(base_path)
    results["training"] = {
        "config": {"d_model": 32, "d_ff": 64, "epochs": 150, "lr": 3e-3,
                   "batch_size": 8, "vocab_size": 300, "seed": 42,
                   "dataset": "examples/datasets/online_learning_base.jsonl (8 examples)",
                   "reasoning": False, "response_only_loss": True},
        "train_time_s": round(train_time, 2),
        "model_size_kb": round(os.path.getsize(base_path) / 1024, 1),
        "param_count": param_count(model),
        "block_size": model.block_size,
    }

    # 2. BEFORE probes.
    results["before"] = probe_all(model, tok)

    # 3. Online learning.
    weights_before = {p: getattr(model, p).copy() for p in model.params}
    t0 = time.time()
    learn_out = model.learn(tok, LEARN_PROMPT, LEARN_RESPONSE,
                            steps=LEARN_STEPS, lr=LEARN_LR, verbose=True)
    learn_time = time.time() - t0
    total_l2 = 0.0
    per_param = {}
    for p in model.params:
        d = float(np.linalg.norm(getattr(model, p) - weights_before[p]))
        per_param[p] = d
        total_l2 += d ** 2
    results["online_learning"] = {
        "learn_time_s": round(learn_time, 2),
        "loss_before": round(learn_out["loss_before"], 4),
        "loss_after": round(learn_out["loss_after"], 4),
        "target_tokens": learn_out["target_tokens"],
        "weight_change_l2_total": round(float(np.sqrt(total_l2)), 6),
        "weight_change_l2_per_param": {k: round(v, 6) for k, v in per_param.items()},
    }

    # 4. AFTER probes.
    results["after"] = probe_all(model, tok)

    # 5. Save + reload + AFTER-RELOAD probes.
    learned_path = os.path.join(WORKDIR, "learned_model.npz")
    model.save(learned_path, tok)
    results["online_learning"]["learned_model_size_kb"] = round(
        os.path.getsize(learned_path) / 1024, 1)
    reloaded, reloaded_tok = MiniGPT.load(learned_path)
    results["after_reload"] = probe_all(reloaded, reloaded_tok)
    # Loss on the learned example after reload (steps=0 => pure evaluation).
    reload_probe = reloaded.learn(reloaded_tok, LEARN_PROMPT, LEARN_RESPONSE,
                                  steps=0)
    results["after_reload"]["learn_example_loss"] = round(
        reload_probe["loss_before"], 4)

    out_path = os.path.join(RESULTS_DIR, "online_learning_poc.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults written to {out_path}")

    # Human-readable summary.
    print("\n================ BEFORE ================")
    for name, q in PROBES:
        print(f"[{name}] {results['before'][name]['reply']!r}")
    print("\n================ AFTER ================")
    for name, q in PROBES:
        print(f"[{name}] {results['after'][name]['reply']!r}")
    print("\n================ AFTER RELOAD ================")
    for name, q in PROBES:
        print(f"[{name}] {results['after_reload'][name]['reply']!r}")
    print("\nlearn_example_loss after reload:",
          results["after_reload"]["learn_example_loss"])


if __name__ == "__main__":
    main()
