"""Dataset abstraction for TAIT."""
from dataclasses import dataclass
from typing import Iterable


@dataclass
class DatasetInfo:
    source: str
    size: int


class DataSource:
    def load(self) -> Iterable[dict]:
        raise NotImplementedError


class LocalJSONLSource(DataSource):
    def __init__(self, path):
        self.path = path

    def load(self):
        from .jsonl import load_jsonl
        rows, _ = load_jsonl(self.path)
        return rows


class HuggingFaceSource(DataSource):
    """Optional adapter. Keeps HF dependencies out of the core."""
    def __init__(self, dataset_id, split="train", **kwargs):
        self.dataset_id, self.split, self.kwargs = dataset_id, split, kwargs

    def load(self):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Hugging Face datasets support requires: pip install 'tait[hf]'") from exc
        return list(load_dataset(self.dataset_id, split=self.split, **self.kwargs))
