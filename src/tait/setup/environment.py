import importlib.util
import json
import os
import platform
from pathlib import Path

STATE_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tait"
STATE_FILE = STATE_DIR / "setup.json"


def is_first_run():
    return not STATE_FILE.exists()


def mark_complete():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"version": 1, "python": platform.python_version()}), encoding="utf-8")


def numpy_available():
    return importlib.util.find_spec("numpy") is not None
