"""TOML configuration loader with CLI-over-config precedence."""
import copy
import tomllib
from pathlib import Path
from .defaults import DEFAULTS


def load_config(path=None, command=None):
    cfg = copy.deepcopy(DEFAULTS.get(command, {}))
    if path:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        raw = raw.get(command, raw)
        cfg.update(raw)
    return cfg


def apply_cli(cfg, args, keys):
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg
