#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash setup-ubuntu.sh first; see README.md for system dependencies." >&2
  exit 1
fi
mode="${1:-play}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  play) exec .venv/bin/python experiments/tick_mjbatch/demo.py --play "$@" ;;
  train) exec .venv/bin/python experiments/tick_mjbatch/demo.py "$@" ;;
  headless) exec .venv/bin/python experiments/tick_mjbatch/demo.py --headless "$@" ;;
  check) exec .venv/bin/python experiments/tick_mjbatch/demo.py --check "$@" ;;
  *) echo "Usage: bash run.sh {play|train|headless|check} [demo options]" >&2; exit 2 ;;
esac
