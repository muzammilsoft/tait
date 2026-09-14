import importlib.util
import os
import platform
import shutil
import sys


def run_doctor():
    print(f"Python: {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"NumPy: {'installed' if importlib.util.find_spec('numpy') else 'missing'}")
    if importlib.util.find_spec('numpy'):
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        try:
            cfg = np.show_config(mode="dicts")
            blas = cfg.get("Build Dependencies", {}).get("blas", {})
            machine = cfg.get("Machine Information", {}).get("host", {})
            simd = cfg.get("SIMD Extensions", {})
            print(f"BLAS: {blas.get('name', '?')} {blas.get('version', '?')}")
            print(f"CPU: {machine.get('cpu', platform.machine())} / {machine.get('system', platform.system())}")
            print(f"SIMD baseline: {simd.get('baseline', [])}")
            name = str(blas.get("name", "")).lower()
            print("[ok] Optimized BLAS detected." if any(k in name for k in ("openblas", "mkl", "blis", "accelerate")) else "[warn] Optimized BLAS not detected.")
        except Exception as exc:
            print(f"[info] NumPy config unavailable: {exc}")
    print(f"Termux: {'yes' if 'PREFIX' in os.environ and 'com.termux' in os.environ.get('PREFIX','') else 'no'}")
    print(f"termux-wake-lock: {'available' if shutil.which('termux-wake-lock') else 'missing'}")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        print(f"{var} = {os.environ.get(var, '(not set)')}")
