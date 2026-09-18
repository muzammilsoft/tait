"""Variant sweep: how aggressive should online learning be?

Loads the trained base model fresh for each variant, runs learn() with
different (steps, lr), and probes. Shows the surgical-vs-destructive
tradeoff honestly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tait.core.generation import generate_reply
from tait.core.model import MiniGPT

WORKDIR = os.path.dirname(os.path.abspath(__file__))
GEN = dict(max_new_tokens=40, temperature=0.7, top_k=8, top_p=0.9, seed=42)
PROBES = [
    ("exact_recall", "What is the capital of the kingdom of Numa?"),
    ("paraphrase_1", "Which city is the capital of Numa?"),
    ("control_1", "What color is grass?"),
    ("forget_A_1", "Which river flows from north to south?"),
]
PROMPT = "What is the capital of the kingdom of Numa?"
RESPONSE = "Zorak is the capital of the kingdom of Numa."

for steps, lr in [(15, 3e-3), (30, 1e-3), (30, 3e-3)]:
    model, tok = MiniGPT.load(os.path.join(WORKDIR, "base_model.npz"))
    out = model.learn(tok, PROMPT, RESPONSE, steps=steps, lr=lr)
    print(f"\n===== steps={steps} lr={lr} "
          f"loss {out['loss_before']:.2f}->{out['loss_after']:.4f} =====")
    for name, q in PROBES:
        print(f"[{name}] {generate_reply(model, tok, q, **GEN)!r}")
