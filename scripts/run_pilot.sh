#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
DATASET="${1:-$ROOT/../Polliradar dataset.zip}"
RUN_DIR="${2:-$ROOT/outputs/pilot}"
python -m qtrc_stop.cli run --config "$ROOT/configs/pilot.yaml" --dataset "$DATASET" --run-dir "$RUN_DIR"
