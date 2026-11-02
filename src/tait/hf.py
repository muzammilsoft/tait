"""Hugging Face publishing for TAIT.

Design rules, kept deliberately small:

* Core TAIT training and chat stay NumPy-only. Publishing is an
  optional path: it imports ``huggingface_hub`` lazily, only when a
  publish is actually run. Install it with ``pip install "tait[publish]"``.
* A token is NEVER printed, logged, or put in a model card, a command
  line, or shell history. The user types it once, hidden, through
  ``tait hf login``; it is stored in a single file with mode 0600.
* The model card is generated from real model metadata (shape, vocab,
  file size, hash, training sidecar when present), never from invented
  claims. Small TAIT models may hallucinate; the card says so.
* The ``.npz`` model format is unchanged, so old TAIT models can be
  published without retraining, and KAIT can read the same file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

TOKEN_ENV = "TAIT_HF_TOKEN"
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "tait"
    return Path.home() / ".config" / "tait"


def token_path() -> Path:
    return config_dir() / "huggingface_token"


def default_repo_path() -> Path:
    return config_dir() / "huggingface_repo"


def read_token() -> str | None:
    """Return the configured token, or None. Never print it."""
    env = os.environ.get(TOKEN_ENV)
    if env and env.strip():
        return env.strip()
    try:
        token = token_path().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return token or None


def save_token(token: str) -> Path:
    """Store a token with mode 0600. Raises ValueError on an empty token."""
    token = token.strip()
    if not token:
        raise ValueError("Empty token: nothing was saved.")
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = token_path()
    # Write first, then restrict, so a partially written file is never
    # left world-readable on a shared filesystem.
    path.write_text(token + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def delete_token() -> bool:
    try:
        token_path().unlink()
        return True
    except FileNotFoundError:
        return False


def save_default_repo(repo_id: str) -> Path:
    validate_repo_id(repo_id)
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = default_repo_path()
    path.write_text(repo_id.strip() + "\n", encoding="utf-8")
    return path


def read_default_repo() -> str | None:
    try:
        repo = default_repo_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return repo or None


def validate_repo_id(repo_id: str) -> str:
    repo_id = (repo_id or "").strip()
    if not _REPO_RE.match(repo_id):
        raise ValueError(
            f"Invalid Hugging Face repo id: {repo_id!r}. "
            "Expected the form 'username/model-name', e.g. 'muzammilsoft/my-tait-model'."
        )
    return repo_id


def model_info(model_path: str | Path) -> dict:
    """Inspect a TAIT .npz file without loading it into a model."""
    import numpy as np

    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"Model file not found: {path}")
    data = np.load(path, allow_pickle=True)
    if "merge_a" not in data.files or "wte" not in data.files:
        raise SystemExit(f"'{path}' is not a valid TAIT model file (missing weights or BPE data).")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha,
        "vocab_size": int(data["wte"].shape[0]),
        "d_model": int(data["wte"].shape[1]),
        "d_ff": int(data["W1"].shape[1]),
        "block_size": int(data["block_size"]),
    }


def load_training_meta(model_path: str | Path) -> dict:
    """Read the <model>.meta.json sidecar written by `tait train`, if any."""
    path = Path(model_path)
    candidates = [
        path.with_suffix(".meta.json"),
        Path(str(path) + ".meta.json"),
    ]
    for candidate in candidates:
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def build_model_card(
    repo_id: str,
    info: dict,
    meta: dict | None = None,
    description: str | None = None,
) -> str:
    meta = meta or {}
    lines = []
    lines.append("---")
    lines.append("license: mit")
    lines.append("library_name: tait")
    lines.append("tags:")
    lines.append("- tait")
    lines.append("- numpy")
    lines.append("- on-device")
    lines.append("- text-generation")
    lines.append("---")
    lines.append("")
    lines.append(f"# {repo_id.split('/', 1)[-1]}")
    lines.append("")
    lines.append(
        description
        or "A small MiniGPT model trained with TAIT, a NumPy-only toolkit "
        "for training AI on phones and resource-constrained devices."
    )
    lines.append("")
    lines.append("## Model details")
    lines.append("")
    lines.append(f"- **Format:** TAIT `.npz` (NumPy weights + BPE tokenizer, readable by the KAIT runner)")
    lines.append(f"- **File:** `{info['file_name']}` ({info['bytes']} bytes, sha256 `{info['sha256'][:16]}...`)")
    lines.append(f"- **Vocab size:** {info['vocab_size']}")
    lines.append(f"- **d_model:** {info['d_model']} | **d_ff:** {info['d_ff']} | **block_size:** {info['block_size']}")
    if meta.get("tait_version"):
        lines.append(f"- **TAIT version:** {meta['tait_version']}")
    if meta.get("data"):
        lines.append(f"- **Training data:** {meta['data']}")
    if meta.get("examples") is not None:
        lines.append(f"- **Training examples:** {meta['examples']}")
    if meta.get("epochs") is not None:
        lines.append(f"- **Epochs:** {meta['epochs']}")
    if meta.get("final_loss") is not None:
        lines.append(f"- **Final training loss:** {meta['final_loss']:.4f}")
    if meta.get("val_loss") is not None:
        lines.append(f"- **Validation loss:** {meta['val_loss']:.4f}")
    lines.append("")
    lines.append("## How to use it")
    lines.append("")
    lines.append("```bash")
    lines.append('pip install "tait[publish]"  # publish support; training itself needs only NumPy')
    lines.append(f"# Download the model, then chat with it locally:")
    lines.append(f"tait chat --model {info['file_name']}")
    lines.append("```")
    lines.append("")
    lines.append("You can also load the same `.npz` file in KAIT, the Android")
    lines.append("runner for TAIT models, by pasting this repository link there.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "This is a tiny model trained from scratch on a small dataset. "
        "It may hallucinate, repeat itself, or answer off-topic. "
        "Treat its output as an experiment, not as a factual source."
    )
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("Trained with TAIT: https://github.com/muzammilsoft/tait")
    lines.append("")
    return "\n".join(lines)


def _require_hub():
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Publishing needs the optional Hugging Face client, which is not installed.\n"
            'Run: pip install "tait[publish]"  (training and chat still need only NumPy)'
        ) from exc
    return HfApi


def publish_model(
    model_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    description: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    token: str | None = None,
) -> dict:
    """Publish a TAIT model to Hugging Face. Returns a summary dict.

    With dry_run=True no network call and no token are needed; the
    model card is generated and returned so it can be inspected.
    """
    repo_id = validate_repo_id(repo_id)
    info = model_info(model_path)
    meta = load_training_meta(model_path)
    card = build_model_card(repo_id, info, meta, description=description)

    if dry_run:
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "model_file": info["file_name"],
            "card": card,
            "meta": meta,
            "dry_run": True,
        }

    token = token or read_token()
    if not token:
        raise SystemExit(
            "No Hugging Face token configured.\n"
            "Run `tait hf login` and paste a fine-grained token with write access "
            "to your own repositories (it is typed hidden and stored with mode 0600). "
            "You can revoke it any time on huggingface.co/settings/tokens."
        )
    api = _require_hub()()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, token=token)
    message = commit_message or f"Publish TAIT model {info['file_name']} (TAIT {meta.get('tait_version', '')})".strip()
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=info["file_name"],
        repo_id=repo_id,
        commit_message=message,
        token=token,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=message,
        token=token,
    )
    if meta:
        api.upload_file(
            path_or_fileobj=json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"),
            path_in_repo="tait-meta.json",
            repo_id=repo_id,
            commit_message=message,
            token=token,
        )
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "model_file": info["file_name"],
        "card": card,
        "meta": meta,
        "dry_run": False,
    }


def whoami(token: str | None = None) -> str | None:
    """Return the HF username for the configured token, or None on failure."""
    token = token or read_token()
    if not token:
        return None
    try:
        api = _require_hub()()
        return api.whoami(token=token).get("name")
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
