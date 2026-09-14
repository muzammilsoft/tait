import shutil
import subprocess
import sys


def is_termux():
    return "com.termux" in __import__("os").environ.get("PREFIX", "")


def install_numpy(noninteractive=False):
    if is_termux() and shutil.which("pkg"):
        cmd = ["pkg", "install", "-y", "python-numpy"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "numpy"]
    print("[setup] Installing NumPy with:", " ".join(cmd))
    if noninteractive:
        return subprocess.call(cmd) == 0
    answer = input("Proceed? [Y/n] ").strip().lower()
    if answer and answer not in ("y", "yes"):
        return False
    return subprocess.call(cmd) == 0
