# Termux

Termux is a first-class target, but it has different Python packaging constraints from desktop Linux.

Recommended baseline:

```bash
pkg update
pkg install python python-numpy
```

TAIT's first-run wizard detects Termux and prefers `pkg install python-numpy` when NumPy is missing. The wizard records completion and does not repeat on normal commands. Use `tait setup --repair` to run it again.

For long training sessions, TAIT uses a best-effort `termux-wake-lock` when available. Android battery optimization can still interrupt workloads.
