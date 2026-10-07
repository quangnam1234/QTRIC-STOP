#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
RUN_DIR="${1:-$ROOT/outputs/full}"
python -m qtrc_stop.cli verify --run-dir "$RUN_DIR"
