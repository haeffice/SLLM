#!/usr/bin/env bash
# Start the BE FastAPI server with BAT inference.
#
# Activate your Python environment (conda) first, then:
#   chmod +x run.sh   # once
#   ./run.sh
#
# Each export uses `${VAR:-default}` so values already set in the shell
# take precedence over the defaults in this file.

set -euo pipefail

# =============================================================================
# 1) BAT 가중치 경로 — 사용자 환경에 맞게 수정 (또는 미리 export)
# =============================================================================
export BAT_LLAMA_PATH="${BAT_LLAMA_PATH:-/path/to/Llama-2-7b-hf}"
export BAT_ENCODER_CKPT="${BAT_ENCODER_CKPT:-/path/to/SpatialAST/finetuned.pth}"
export BAT_PROJECTOR_CKPT="${BAT_PROJECTOR_CKPT:-/path/to/BAT/model.pt}"

# =============================================================================
# 2) (선택) 모델 / 장치
# =============================================================================
export BAT_DEVICE="${BAT_DEVICE:-cuda:0}"            # cuda:N 또는 cpu
export AUDIO_LLM_ENABLED="${AUDIO_LLM_ENABLED:-bat}" # 콤마 구분으로 추가 모델 등록
export AUDIO_LLM_DEFAULT="${AUDIO_LLM_DEFAULT:-bat}" # /inference?model= 미지정 시 기본

# =============================================================================
# 3) 서버 바인딩
# =============================================================================
HOST="${BE_HOST:-0.0.0.0}"
PORT="${BE_PORT:-9001}"

# =============================================================================
# 4) (이하 수정 불필요)
# =============================================================================
BE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BE_DIR"

# --- 가중치 경로 사전 점검 ----------------------------------------------------
echo "==> resolved checkpoint paths"
missing=0
for var in BAT_LLAMA_PATH BAT_ENCODER_CKPT BAT_PROJECTOR_CKPT; do
    val="${!var}"
    if [[ ! -e "$val" ]]; then
        echo "    [MISSING] $var = $val"
        missing=1
    else
        echo "    [ok]      $var = $val"
    fi
done
if (( missing )); then
    echo "==> WARNING: 위 경로가 없으면 BAT 모델 로드가 실패하고 /health의 status=failed로 노출됩니다."
    echo "    (서버 자체는 기동되어 /localize는 정상 동작)"
fi

# --- run ---------------------------------------------------------------------
echo "==> AUDIO_LLM_ENABLED=$AUDIO_LLM_ENABLED  AUDIO_LLM_DEFAULT=$AUDIO_LLM_DEFAULT  BAT_DEVICE=$BAT_DEVICE"
echo "==> starting uvicorn on ${HOST}:${PORT}"
exec uvicorn main:app --host "$HOST" --port "$PORT" "$@"
