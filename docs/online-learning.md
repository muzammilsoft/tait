# Online Learning in TAIT

TAIT models can update their weights at inference time from a single new
example, without re-running full training. This is exposed as one method:

```python
model.learn(tokenizer, prompt, response, steps=10, lr=3e-3)
```

## What it is

`MiniGPT.learn()` encodes one `(prompt, response)` pair exactly like
training does (`format_example` serialization, SEP/END framing,
response-only loss mask) and then runs `steps` forward → loss →
backward → Adam steps on that single example, reusing the model's own
optimizer state. Weights are mutated in place. Nothing is saved
automatically — call `model.save()` yourself.

It returns a dict with `loss_before`, `loss_after`, per-step `losses`,
`steps`, `lr`, and `target_tokens`, so experiments can log what happened.

## How it differs from traditional training

| Traditional training (`tait train`) | Online learning (`model.learn()`) |
|---|---|
| Whole dataset, many epochs | One example, a few steps |
| Train/validation split, early stopping | No validation, no early stopping |
| BPE trained on the dataset | Reuses the loaded tokenizer as-is |
| Checkpoints written automatically | Nothing saved unless you call `save()` |
| Goal: general competence | Goal: memorize one new fact *now* |

## What it is NOT

- **Online learning ≠ AGI.** It is gradient descent on one example.
- **Online learning ≠ guaranteed reasoning.** The model does not
  understand the fact; it adjusts weights to reproduce the token sequence.
- **Online learning ≠ automatic self-improvement.** There is no
  self-evaluation, no curriculum, no memory system — just weight updates.
- **No RAG, no vector DB, no agents.** This feature is a research hook
  for studying continual weight updates, nothing more.

## Limitations (observed in the proof-of-concept experiment)

With the tiny default MiniGPT (1 layer, 1 head, ~29K params):

1. **Memorization, not generalization.** After learning, the exact
   training phrasing reproduces the fact's beginning, but paraphrases do
   not retrieve it.
2. **Repetition collapse.** Pushing the example loss to ~0 overfits the
   single sequence; generation after the fact's head degenerates into
   repeated fragments.
3. **Catastrophic interference.** Even a modest update (15 steps)
   visibly degrades previously learned answers (controls) and other
   memorized facts. The small model cannot isolate new knowledge.
4. **No protection against bad examples.** Whatever you pass to
   `learn()` is baked into the weights. There is no undo besides
   reloading a checkpoint.

Practical guidance: keep `steps` small, keep `lr` at or below the
training learning rate, and always keep a checkpoint from before
learning so you can roll back.

## Reproducing the experiment

```bash
python experiments/online_learning_poc.py
```

Trains a small base model on `examples/datasets/online_learning_base.jsonl`,
probes it before/after learning one fictional fact
(`Zorak is the capital of the kingdom of Numa`), saves, reloads, and
writes everything to `experiments/results/online_learning_poc.json`.
`experiments/learn_sweep.py` compares (steps, lr) settings.
