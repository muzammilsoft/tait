# Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
python -m pytest -q
python -m tait doctor
```

The package lives under `src/tait`. Keep heavyweight dependencies out of the core. New dataset integrations should implement `DataSource`; model additions should not require CLI rewrites.
