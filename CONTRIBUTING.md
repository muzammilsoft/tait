# Contributing to TAIT

Thank you for contributing. TAIT is designed to stay lightweight and useful on Termux and constrained devices.

## Before a PR

Run:

```bash
python -m pytest -q
python -m tait doctor
```

For model or CLI changes, also run the demo training smoke test described in `docs/release-process.md`.

## Principles

- Keep the base dependency footprint small.
- Preserve Python/Termux compatibility.
- Do not hide network access in core commands.
- Keep features modular and independently testable.
- Document behavior changes.

Read `docs/coding-guidelines.md` and `docs/architecture.md` before changing core boundaries.
