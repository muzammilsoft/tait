"""Lightweight byte-level BPE tokenizer used by TAIT."""
from collections import Counter, defaultdict

import numpy as np

SEP, END = "\x01", "\x02"


def progress_bar(current, total, label="", width=16, newline_at_end=True):
    total = max(total, 1)
    frac = min(current / total, 1.0)
    filled = int(round(width * frac))
    bar = "\u25c6" * filled + "\u25c7" * (width - filled)
    pct = int(frac * 100)
    end = "\n" if (newline_at_end and current >= total) else ""
    print(f"\r{bar} {pct:3d}% ({label})", end=end, flush=True)


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.sep_id = self.end_id = self.pad_id = None

    def train(self, texts, vocab_size=800, sample_size=3000, seed=0, show_progress=True):
        if vocab_size < 259:
            raise ValueError("vocab_size must be at least 259")
        rng = np.random.default_rng(seed)
        texts = list(texts)
        if len(texts) > sample_size:
            idx = rng.choice(len(texts), sample_size, replace=False)
            texts = [texts[i] for i in idx]
        num_merges = vocab_size - 256
        ids_list = [list(t.encode("utf-8")) for t in texts]
        stats, pair_to_seqs = Counter(), defaultdict(set)
        for si, ids in enumerate(ids_list):
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                stats[pair] += 1
                pair_to_seqs[pair].add(si)
        for m in range(num_merges):
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + m
            affected = list(pair_to_seqs.get(pair, ()))
            for si in affected:
                ids = ids_list[si]
                for i in range(len(ids) - 1):
                    p = (ids[i], ids[i + 1])
                    stats[p] -= 1
                    if stats[p] <= 0:
                        del stats[p]
                        pair_to_seqs.pop(p, None)
            for si in affected:
                ids_list[si] = self._merge(ids_list[si], pair, new_id)
            for si in affected:
                ids = ids_list[si]
                for i in range(len(ids) - 1):
                    p = (ids[i], ids[i + 1])
                    stats[p] += 1
                    pair_to_seqs[p].add(si)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if show_progress and (m % max(1, num_merges // 100) == 0 or m == num_merges - 1):
                progress_bar(m + 1, num_merges, "BPE training", newline_at_end=(m == num_merges - 1))
        base = 256 + len(self.merges)
        self.sep_id, self.end_id, self.pad_id = base, base + 1, base + 2

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    @property
    def vocab_size_total(self):
        return 256 + len(self.merges) + 3

    @staticmethod
    def _get_stats(ids):
        c = Counter()
        for i in range(len(ids) - 1):
            c[(ids[i], ids[i + 1])] += 1
        return c

    def _encode_chunk(self, ids):
        ids = list(ids)
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            candidates = [p for p in stats if p in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        if self.sep_id is None:
            raise RuntimeError("Tokenizer is not trained/loaded")
        result, buf = [], ""
        for ch in text:
            if ch in (SEP, END):
                if buf:
                    result.extend(self._encode_chunk(list(buf.encode("utf-8"))))
                    buf = ""
                result.append(self.sep_id if ch == SEP else self.end_id)
            else:
                buf += ch
        if buf:
            result.extend(self._encode_chunk(list(buf.encode("utf-8"))))
        return result

    def decode(self, ids):
        if self.sep_id is None:
            raise RuntimeError("Tokenizer is not trained/loaded")
        out, parts = b"", []
        for i in ids:
            if i == self.sep_id:
                parts.append(out.decode("utf-8", errors="replace")); parts.append(SEP); out = b""
            elif i == self.end_id:
                parts.append(out.decode("utf-8", errors="replace")); parts.append(END); out = b""
            elif i == self.pad_id:
                continue
            else:
                out += self.vocab.get(int(i), b"")
        parts.append(out.decode("utf-8", errors="replace"))
        return "".join(parts)
