import os
import stat

import numpy as np
import pytest

from tait.core.model import MiniGPT
from tait.core.tokenizer import BPETokenizer
from tait.hf import (
    build_model_card,
    load_training_meta,
    model_info,
    publish_model,
    save_token,
    validate_repo_id,
)


def _tiny_model(tmp_path):
    tok = BPETokenizer()
    tok.train(["hello world", "hello tait", "hello small model"], vocab_size=300, sample_size=100)
    model = MiniGPT(tok.vocab_size_total, pad_id=tok.pad_id, d_model=8, d_ff=16, block_size=8, seed=42)
    path = tmp_path / "tiny.npz"
    model.save(str(path), tok)
    return path


def test_repo_id_validation():
    assert validate_repo_id("muzammilsoft/my-model") == "muzammilsoft/my-model"
    with pytest.raises(ValueError):
        validate_repo_id("just-a-name")
    with pytest.raises(ValueError):
        validate_repo_id("user/model with spaces")


def test_model_card_from_real_model(tmp_path):
    path = _tiny_model(tmp_path)
    info = model_info(path)
    assert info["vocab_size"] > 0 and info["d_model"] == 8
    card = build_model_card("muzammilsoft/tiny-test", info, {"tait_version": "2.3.0", "final_loss": 1.2345})
    assert "TAIT" in card
    assert "2.3.0" in card
    assert "1.2345" in card
    assert "hallucinate" in card.lower()
    assert "hf_" not in card  # no token-like material in a card


def test_dry_run_uploads_nothing(tmp_path, monkeypatch):
    path = _tiny_model(tmp_path)
    monkeypatch.delenv("TAIT_HF_TOKEN", raising=False)
    result = publish_model(path, "muzammilsoft/tiny-test", dry_run=True)
    assert result["dry_run"] is True
    assert result["url"] == "https://huggingface.co/muzammilsoft/tiny-test"
    assert "MiniGPT" in result["card"]


def test_publish_without_token_fails_cleanly(tmp_path, monkeypatch):
    path = _tiny_model(tmp_path)
    monkeypatch.delenv("TAIT_HF_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    with pytest.raises(SystemExit) as exc:
        publish_model(path, "muzammilsoft/tiny-test")
    assert "tait hf login" in str(exc.value)
    assert "hf_" not in str(exc.value)


def test_token_file_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    path = save_token("hf_test_token_value_never_printed")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600
    assert path.read_text().strip() == "hf_test_token_value_never_printed"


def test_training_meta_sidecar(tmp_path):
    meta_path = tmp_path / "model.meta.json"
    meta_path.write_text('{"tait_version": "2.3.0", "final_loss": 2.5}', encoding="utf-8")
    model_path = tmp_path / "model.npz"
    assert load_training_meta(model_path)["final_loss"] == 2.5
    assert load_training_meta(tmp_path / "missing.npz") == {}
