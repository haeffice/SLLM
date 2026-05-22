#!/usr/bin/env bash
# V-JEPA 2 frozen-encoder attentive-probe evaluation launcher.
#
# Usage:
#   ./run_eval_VJEPA2.sh config.yaml student.pt
#
# Args:
#   $1  config.yaml  — SAME config used for pre-training (model section
#                      rebuilds the encoder; an `eval:` section supplies the
#                      labelled manifests + num_classes).
#   $2  student.pt   — student checkpoint from training
#                      (vjepa2_student_step*.pt).

set -euo pipefail

CONFIG="${1:?usage: run_eval_VJEPA2.sh config.yaml student.pt}"
CKPT="${2:?usage: run_eval_VJEPA2.sh config.yaml student.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[[ -f "$CKPT"   ]] || { echo "ERROR: ckpt not found: $CKPT"   >&2; exit 1; }

echo "==> python eval_vjepa2.py --config $CONFIG --ckpt $CKPT"
exec python eval_vjepa2.py --config "$CONFIG" --ckpt "$CKPT" "${@:3}"
