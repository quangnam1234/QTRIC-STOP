#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
STAGE="${1:?Usage: run_stage.sh cp0|cp1|cp2|cp3|cp4|cp5|cp6|cp8 [dataset] [run_dir] [config]}"
DATASET="${2:-$ROOT/../Polliradar dataset.zip}"
RUN_DIR="${3:-$ROOT/outputs/full}"
CONFIG="${4:-$ROOT/configs/full.yaml}"
python -m qtrc_stop.cli run --config "$CONFIG" --dataset "$DATASET" --run-dir "$RUN_DIR" --start-stage "$STAGE" --end-stage "$STAGE"
