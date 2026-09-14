import os


def progress_bar(current, total, label="", width=16, newline_at_end=True):
    total = max(total, 1)
    frac = min(current / total, 1.0)
    filled = int(round(width * frac))
    bar = "\u25c6" * filled + "\u25c7" * (width - filled)
    pct = int(frac * 100)
    end = "\n" if (newline_at_end and current >= total) else ""
    print(f"\r{bar} {pct:3d}% ({label})", end=end, flush=True)


def print_banner():
    print("TAIT - Termux AI Training | Made in Sudan 🇸🇩")
