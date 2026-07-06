#!/usr/bin/env bash
# JEPA Neural Tokenizer evaluation launcher (round-trip reconstruction).
#
# Usage:
#   ./run_eval_NeuralTokenizer.sh config.yaml nt_stage2_step29000.pt
#
# Args:
#   $1  config.yaml  — SAME config used for training (model section rebuilds
#                      the tokenizer; an `eval:` section supplies the manifest).
#   $2  nt_stage2_*.pt — Stage-2 checkpoint (encoder + FSQ proj + decoder).
#   $3+ forwarded to eval_neural_tokenizer.py (e.g. --batch-size 4).

set -euo pipefail

CONFIG="${1:?usage: run_eval_NeuralTokenizer.sh config.yaml nt_stage2.pt}"
CKPT="${2:?usage: run_eval_NeuralTokenizer.sh config.yaml nt_stage2.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"   >&2; exit 1; }

echo "==> python eval_neural_tokenizer.py --config $CONFIG --ckpt $CKPT ${*:3}"
exec python eval_neural_tokenizer.py --config "$CONFIG" --ckpt "$CKPT" "${@:3}"
