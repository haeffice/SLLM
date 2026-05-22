#!/usr/bin/env bash
# BAT 모델을 torchrun으로 분산 실행해 SpatialSoundQA eval JSON을 처리.
#
# 사용 방법
#   1) 아래 PATHS 블록의 값을 환경에 맞게 채운다
#   2) (필요 시) 활성화할 conda env로 들어간다
#   3) ./run_eval_BAT.sh
#
# 환경변수로 디폴트를 덮어쓰는 것도 가능:
#   NPROC_PER_NODE=4 BATCH_SIZE=8 ./run_eval_BAT.sh

set -euo pipefail

# =============================================================================
# 1) PATHS — 사용자 환경에 맞게 채워야 함
# =============================================================================
# eval 데이터셋 JSON ({"data": [...]}) 경로
EVAL_JSON=""

# audio_id 의 기준 디렉터리.  audio_path = ${AUDIO_ROOT}/${audio_id}
AUDIO_ROOT=""

# reverb_id 의 기준 디렉터리.  reverb_path = ${REVERB_ROOT}/${reverb_id}
REVERB_ROOT=""

# 예측 결과를 저장할 디렉터리.  최종 출력은 ${OUTPUT_DIR}/predictions.jsonl
OUTPUT_DIR=""

# BAT 가중치 경로
LLAMA_PATH=""        # local Llama-2-7b-hf checkpoint
ENCODER_CKPT=""      # SpatialAST finetuned.pth
PROJECTOR_CKPT=""    # BAT model.pt (Q-Former + LoRA)

# =============================================================================
# 2) torchrun / inference 옵션 (필요 시 조정)
# =============================================================================
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"   # 사용할 GPU 수
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"

# =============================================================================
# 3) Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 경로 사전 검증 (없으면 즉시 종료)
for var in EVAL_JSON AUDIO_ROOT REVERB_ROOT OUTPUT_DIR LLAMA_PATH ENCODER_CKPT PROJECTOR_CKPT; do
    val="${!var}"
    if [[ -z "$val" ]]; then
        echo "ERROR: $var is empty — edit the PATHS section in run_eval_BAT.sh first." >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "==> torchrun --nproc_per_node=$NPROC_PER_NODE BAT.py"
echo "    eval_json   = $EVAL_JSON"
echo "    audio_root  = $AUDIO_ROOT"
echo "    reverb_root = $REVERB_ROOT"
echo "    output_dir  = $OUTPUT_DIR"
echo "    batch_size  = $BATCH_SIZE, num_workers=$NUM_WORKERS"
echo "    num_beams   = $NUM_BEAMS, max_new_tokens=$MAX_NEW_TOKENS"

exec torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    BAT.py \
    --eval-json "$EVAL_JSON" \
    --audio-root "$AUDIO_ROOT" \
    --reverb-root "$REVERB_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --llama-path "$LLAMA_PATH" \
    --encoder-ckpt "$ENCODER_CKPT" \
    --projector-ckpt "$PROJECTOR_CKPT" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-beams "$NUM_BEAMS" \
    --max-new-tokens "$MAX_NEW_TOKENS"
