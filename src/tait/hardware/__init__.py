from .doctor import run_doctor
from .benchmark import run_benchmark
from .termux import is_termux, wake_lock_acquire, wake_lock_release

__all__ = ["run_doctor", "run_benchmark", "is_termux", "wake_lock_acquire", "wake_lock_release"]
