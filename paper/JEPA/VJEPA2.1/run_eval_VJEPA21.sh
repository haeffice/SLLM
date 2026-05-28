#!/usr/bin/env bash
# V-JEPA 2.1 frozen-encoder evaluation launcher.
#
# Usage:
#   ./run_eval_VJEPA21.sh config.yaml student.pt                 # attentive probe
#   ./run_eval_VJEPA21.sh config.yaml student.pt --mode dense    # dense feature export
#
# Args:
#   $1  config.yaml  — SAME config used for pre-training (model section
#                      rebuilds the encoder; an `eval:` section supplies the
#                      labelled manifests + num_classes).
#   $2  student.pt   — student checkpoint (vjepa21_student_step*.pt).
#   $3+ forwarded to eval_vjepa21.py (e.g. --mode dense --dense-out feats.npy).

set -euo pipefail

CONFIG="${1:?usage: run_eval_VJEPA21.sh config.yaml student.pt}"
CKPT="${2:?usage: run_eval_VJEPA21.sh config.yaml student.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"   >&2; exit 1; }

echo "==> python eval_vjepa21.py --config $CONFIG --ckpt $CKPT ${*:3}"
exec python eval_vjepa21.py --config "$CONFIG" --ckpt "$CKPT" "${@:3}"
