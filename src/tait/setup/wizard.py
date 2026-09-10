from .dependencies import install_numpy
from .environment import is_first_run, mark_complete, numpy_available
from ..hardware.termux import is_termux


def run_setup(repair=False, yes=False):
    first = is_first_run()
    if not first and not repair:
        print("TAIT setup is already complete. Use 'tait setup --repair' to run it again.")
        return 0
    print("TAIT setup")
    print("===========")
    print(f"Environment: {'Termux' if is_termux() else 'Linux/other Python'}")
    if not numpy_available():
        if not install_numpy(noninteractive=yes):
            print("[error] NumPy is required. Setup not completed.")
            return 1
    else:
        print("[ok] NumPy is available")
    mark_complete()
    print("[ok] Setup complete")
    return 0
