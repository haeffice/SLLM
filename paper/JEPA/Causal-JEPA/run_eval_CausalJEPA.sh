#!/usr/bin/env bash
# Causal-JEPA latent-rollout / planning evaluation launcher.
#
# Usage:
#   ./run_eval_CausalJEPA.sh config.yaml causal_jepa_step2000.pt
#
# Args:
#   $1  config.yaml  — SAME config used for pre-training (model section
#                      rebuilds the network; an `eval:` section supplies the
#                      held-out manifest).
#   $2  model.pt     — model checkpoint (causal_jepa_step*.pt).
#   $3+ forwarded to eval_causal_jepa.py (e.g. --batch-size 8).

set -euo pipefail

CONFIG="${1:?usage: run_eval_CausalJEPA.sh config.yaml model.pt}"
CKPT="${2:?usage: run_eval_CausalJEPA.sh config.yaml model.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"   >&2; exit 1; }

echo "==> python eval_causal_jepa.py --config $CONFIG --ckpt $CKPT ${*:3}"
exec python eval_causal_jepa.py --config "$CONFIG" --ckpt "$CKPT" "${@:3}"
