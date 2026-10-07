#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
DATASET="${1:-$ROOT/../Polliradar dataset.zip}"
RUN_DIR="${2:-$ROOT/outputs/full}"
BASE_CONFIG="${3:-$ROOT/configs/full.yaml}"
TMP_CONFIG="$RUN_DIR/robustness_enabled.yaml"
python - "$BASE_CONFIG" "$TMP_CONFIG" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('robustness', {})['enabled'] = True
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
PY
python -m qtrc_stop.cli run --config "$TMP_CONFIG" --dataset "$DATASET" --run-dir "$RUN_DIR" --start-stage cp7 --end-stage cp7
