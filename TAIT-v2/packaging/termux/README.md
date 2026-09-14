# Termux packaging

The canonical application code is the Python package under `src/tait`. Termux packaging should wrap that package rather than duplicate it.

The preferred runtime dependency on Termux is the system `python-numpy` package.
