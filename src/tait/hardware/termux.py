import os
import shutil


def is_termux():
    prefix = os.environ.get("PREFIX", "")
    return "com.termux" in prefix or bool(shutil.which("termux-info"))


def wake_lock_acquire():
    if shutil.which("termux-wake-lock"):
        ok = os.system("termux-wake-lock >/dev/null 2>&1") == 0
        print("[lock] termux-wake-lock acquired" if ok else "[lock] failed to acquire wake lock")
    else:
        print("[lock] termux-wake-lock unavailable; continuing")


def wake_lock_release():
    if shutil.which("termux-wake-unlock"):
        os.system("termux-wake-unlock >/dev/null 2>&1")
