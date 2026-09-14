#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
# Template for a future Termux repository package. The canonical Python source stays in src/tait.
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
echo "Build TAIT from the repository with the Termux packaging system."
echo "Runtime dependency: python, python-numpy"
echo "Canonical entry point: tait.cli.main:main"
echo "PREFIX=$PREFIX"
