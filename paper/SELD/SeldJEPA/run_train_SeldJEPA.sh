#!/usr/bin/env bash
# SeldJEPA Stage-1 — causal-Conformer LeJEPA pre-training launcher.
#
# Usage:
#   ./run_train_SeldJEPA.sh config_pretrain.yaml
#   NPROC_PER_NODE=4 ./run_train_SeldJEPA.sh config_pretrain.yaml
#
# Takes EXACTLY one argument: the path to config_pretrain.yaml. Refuses to start
# if the configured output_dir already contains checkpoints (HF `checkpoint-*/`
# dirs or `*.pt` model dumps) so an existing run is never silently overwritten.

set -euo pipefail

CONFIG="${1:?usage: run_train_SeldJEPA.sh config_pretrain.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

OUTPUT_DIR="$(python -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['train']['output_dir'])" "$CONFIG")"
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: train.output_dir is empty in $CONFIG" >&2
    exit 1
fi

if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null 2>&1 \
   || compgen -G "$OUTPUT_DIR"/*.pt > /dev/null 2>&1; then
    echo "ERROR: checkpoints already exist in $OUTPUT_DIR — refusing to start." >&2
    echo "       Move/delete them or point train.output_dir elsewhere." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "==> torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train_seld_jepa.py"
echo "    config     = $CONFIG"
echo "    output_dir = $OUTPUT_DIR"

exec torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    train_seld_jepa.py \
    --config "$CONFIG"
