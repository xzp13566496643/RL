#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This package targets Ubuntu Linux amd64 / x86_64." >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Use Python 3.11, 3.12 or 3.13 (Ubuntu 24.04: python3.12)."'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ./vendor/mjbatch
printf '\nInstallation finished. Open viewer: bash run.sh play --seconds 30\n'
