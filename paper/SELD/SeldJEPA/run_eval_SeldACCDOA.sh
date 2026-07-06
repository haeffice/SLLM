#!/usr/bin/env bash
# SeldJEPA Stage-2 — Multi-ACCDOA SELD evaluation launcher.
#
# Usage:
#   ./run_eval_SeldACCDOA.sh config_seld.yaml seld_accdoa_final.pt
#
# Args:
#   $1  config_seld.yaml  — SAME config used for training (rebuilds features;
#                           an `eval:` section supplies the held-out manifest).
#   $2  model.pt          — Stage-2 checkpoint (seld_accdoa_step*.pt / _final.pt).
#   $3+ forwarded to eval_seld_accdoa.py (e.g. --manifest data/test.json --batch-size 8).

set -euo pipefail

CONFIG="${1:?usage: run_eval_SeldACCDOA.sh config_seld.yaml model.pt}"
CKPT="${2:?usage: run_eval_SeldACCDOA.sh config_seld.yaml model.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"   >&2; exit 1; }

echo "==> python eval_seld_accdoa.py --config $CONFIG --ckpt $CKPT ${*:3}"
exec python eval_seld_accdoa.py --config "$CONFIG" --ckpt "$CKPT" "${@:3}"
