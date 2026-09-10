from pathlib import Path
from tait.data.jsonl import load_jsonl
from tait.config.loader import load_config


def test_jsonl_aliases_and_invalid(tmp_path):
    p = Path(tmp_path) / "d.jsonl"
    p.write_text('{"question":"q","answer":"a"}\nnot json\n{"prompt":"x"}\n', encoding="utf-8")
    rows, invalid = load_jsonl(p)
    assert rows == [{"prompt":"q","response":"a"}]
    assert invalid == 2


def test_config_loads_toml(tmp_path):
    p = Path(tmp_path) / "c.toml"
    p.write_text('[train]\nepochs = 2\nbatch_size = 3\n', encoding="utf-8")
    c = load_config(p, "train")
    assert c["epochs"] == 2 and c["batch_size"] == 3
