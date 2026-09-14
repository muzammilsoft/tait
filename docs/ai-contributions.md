# AI Contributions

TAIT has been developed collaboratively with AI coding systems and human direction.

## Project provenance

- **Claude:** initial single-file MiniGPT engine, BPE tokenizer, training pipeline, checkpointing, and original CLI.
- **GPT:** v2 modular architecture, package layout, configuration direction, setup wizard design, data abstraction, local Web Chat, documentation structure, tests, and refactoring of the original engine into package boundaries.
- **TAIT 2.1 work:** compact reasoning/CoT serialization, response-only loss, validation/early stopping, safe interrupt checkpoints, and example dataset design were added while preserving the v2 model/checkpoint format.

This file exists so future maintainers can distinguish inherited components from later architectural changes and review them deliberately. AI-generated code is subject to the same tests and human review as any other contribution.
