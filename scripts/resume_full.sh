#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
DATASET="${1:-$ROOT/../Polliradar dataset.zip}"
RUN_DIR="${2:-$ROOT/outputs/full}"
# Rerun the same command. Completed per-recording and per-fold tasks are skipped.
python -m qtrc_stop.cli run --config "$ROOT/configs/full.yaml" --dataset "$DATASET" --run-dir "$RUN_DIR"
