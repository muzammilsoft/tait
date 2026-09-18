# AI Contributions

TAIT has been developed collaboratively with AI coding systems and human direction.

## Project provenance

- **Claude:** initial single-file MiniGPT engine, BPE tokenizer, training pipeline, checkpointing, and original CLI.
- **GPT:** v2 modular architecture, package layout, configuration direction, setup wizard design, data abstraction, local Web Chat, documentation structure, tests, and refactoring of the original engine into package boundaries.
- **TAIT 2.1 work:** compact reasoning/CoT serialization, response-only loss, validation/early stopping, safe interrupt checkpoints, and example dataset design were added while preserving the v2 model/checkpoint format.

This file exists so future maintainers can distinguish inherited components from later architectural changes and review them deliberately. AI-generated code is subject to the same tests and human review as any other contribution.

## Agent change log

### 2026-09-18 — Muse: Online Learning / inference-time learning

- **Agent:** Muse. **Date:** 2026-09-18. **Branch:** `feature/online-learning`.
- **Goal:** let a TAIT model learn from one new example at inference time
  by updating its weights, without a full retraining run.

**Files changed and why:**

- `src/tait/core/model.py` — added `MiniGPT.learn(tokenizer, prompt,
  response, ...)` plus a private `_single_example_batch()` helper. The
  method encodes the example exactly like training
  (`format_example` + SEP/END, response-only loss mask) and runs
  `steps` forward/backward/Adam steps reusing the model's own optimizer
  state. Returns `{loss_before, loss_after, losses, steps, lr,
  target_tokens}`. Deliberately a method on MiniGPT (no new abstraction,
  no new module): the user asked for a `model.learn(...)`-style API and
  this keeps the change surface minimal.
- `tests/test_online_learning.py` — 8 tests: learn executes and reports
  losses; weights really change; loss decreases; **generation output
  changes and contains the learned fact** (behavioral); `steps=0` is a
  no-op; save→reload preserves the learning; traditional training still
  works end-to-end via the real `cmd_train` path; inference is
  deterministic without learning.
- `examples/datasets/online_learning_base.jsonl` — 8 base examples
  (general knowledge + fictional "fact A": the river Sila).
- `examples/datasets/online_learning_facts.jsonl` — 6 fictional facts
  for online-learning experiments (fact B: Zorak/Numa, Qirb bird,
  Captain Rava, daru currency, Mount Thar, Sila's end).
- `experiments/online_learning_poc.py` — reproducible end-to-end run:
  train → probe BEFORE → learn → probe AFTER → save → reload → probe
  AFTER-RELOAD; writes `experiments/results/online_learning_poc.json`.
  Fixed generation settings (temp 0.7, top_k 8, top_p 0.9, 40 tokens,
  seed 42) for fair comparison.
- `experiments/learn_sweep.py` — compares (steps, lr) variants.
- `experiments/pilot_train.py` — scratch pilot; safe to delete.
- `docs/online-learning.md` — new doc: what the feature is, how it
  differs from training, explicit "≠ AGI / ≠ reasoning / ≠
  self-improvement" disclaimers, observed limitations, how to reproduce.
- `.github/workflows/publish.yml` — recreated PyPI publish workflow
  (build → twine check → publish on GitHub Release). Uses the existing
  `PYPI_API_TOKEN` repository secret by name only; no secret in the repo.

**Design decisions:**

- Single-example API; callers loop for multiple examples.
- No new dependencies, no architecture change, no CLI command (internal
  API + tests + reproducible experiment were the priority; a `tait learn`
  CLI can be added later if wanted).
- Optimizer state is still not saved in checkpoints (pre-existing
  limitation); learn() continues from in-memory Adam state, which is
  correct for continued optimization.
- Checkpoint format unchanged — old checkpoints load fine.

**Test results:** 15/15 pass (7 pre-existing + 8 new).

**Real experiment results (honest, unembellished):**

- Base: d_model=32, d_ff=64, 8 examples, 150 epochs, 1.5s, 29,327
  params, 191.3 KB, final loss 0.0016 (Python 3.12.3, numpy 2.5.3).
- learn(): 100 steps, lr=3e-3, 0.2s, loss 12.9214 → 0.0014, total
  weight L2 change 7.92 (largest in Whead: 6.70).
- **Exact recall:** partial — reply begins "Zorak is the ..." then
  degenerates into repetition ("cak is cak ..."). The fact's head is
  memorized; generation dynamics break after a few tokens.
- **Paraphrase:** failed — paraphrased questions produce garbage with
  fragments of the learned fact. No generalization.
- **Control:** damaged — previously correct answers ("The grass is
  green.") become garbage. Even 15 steps cause visible interference.
- **Catastrophic forgetting:** confirmed — fact A ("The river Sila
  flows from north to south.") is destroyed after learning fact B, at
  every (steps, lr) setting tried: (15, 3e-3), (30, 1e-3), (30, 3e-3),
  (100, 3e-3).
- **Save/reload:** learning persists exactly (loss 0.0014 after reload,
  identical replies with fixed seed).

**What did NOT work / open problems:**

- The tiny 1-layer model cannot isolate new knowledge; any learn() run
  tested so far trades the new fact against global generation quality.
- No undo except reloading a pre-learning checkpoint — keep one.
- learn() examples longer than block_size+1 are truncated (same as
  training); documented but not specially handled.

**TODOs for the next agent:**

- Consider a gentler default (fewer steps / lower lr) and document the
  steps×lr tradeoff curve more finely.
- Possible research directions: elastic weight consolidation (EWC),
  rehearsal (mix old examples into learn()), or LoRA-style adapters so
  the base weights stay frozen.
- Add `tait learn` CLI only if a concrete use case demands it.
- Delete `experiments/pilot_train.py` once the PoC script is accepted.
