#!/usr/bin/env bash
# Start the BE FastAPI translation server over HTTPS.
#
# Activate your Python environment first, then:
#   chmod +x run.sh        # once
#   bash gen-cert.sh <LAN_IP>   # once — creates cert.pem / key.pem
#   ./run.sh /path/to/ckpt      # checkpoint path (optional for the mock)
#
# The first positional arg, if given, is exported as <DEFAULT_MODEL>_CKPT.
# Each export uses `${VAR:-default}` so values already set in the shell take
# precedence over the defaults in this file.

set -euo pipefail

# =============================================================================
# 1) 모델 / 장치
# =============================================================================
export AUDIO_LLM_ENABLED="${AUDIO_LLM_ENABLED:-mock}"  # 콤마 구분으로 추가 모델 등록
export AUDIO_LLM_DEFAULT="${AUDIO_LLM_DEFAULT:-mock}"   # /translate?model= 미지정 시 기본
export MOCK_DEVICE="${MOCK_DEVICE:-cpu}"                # cuda:N 또는 cpu

# 체크포인트 경로: 첫 번째 위치 인자 → <기본모델>_CKPT 로 export (편의 기능).
# 예) ./run.sh /path/to/ckpt  →  MOCK_CKPT=/path/to/ckpt
if [[ $# -gt 0 && "$1" != -* ]]; then
    CKPT_VAR="$(echo "$AUDIO_LLM_DEFAULT" | tr '[:lower:]' '[:upper:]')_CKPT"
    export "$CKPT_VAR=$1"
    echo "==> $CKPT_VAR=$1"
    shift
fi

# =============================================================================
# 2) 서버 바인딩 / TLS
# =============================================================================
HOST="${BE_HOST:-0.0.0.0}"
PORT="${BE_PORT:-9001}"

BE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BE_DIR"

if [[ ! -f cert.pem || ! -f key.pem ]]; then
    echo "==> ERROR: cert.pem / key.pem 이 없습니다."
    echo "    먼저 자체서명 인증서를 생성하세요:  bash gen-cert.sh <LAN_IP>"
    exit 1
fi

# =============================================================================
# 3) run
# =============================================================================
echo "==> AUDIO_LLM_ENABLED=$AUDIO_LLM_ENABLED  AUDIO_LLM_DEFAULT=$AUDIO_LLM_DEFAULT  MOCK_DEVICE=$MOCK_DEVICE"
echo "==> starting HTTPS uvicorn on ${HOST}:${PORT}"
exec uvicorn main:app --host "$HOST" --port "$PORT" \
  --ssl-keyfile key.pem --ssl-certfile cert.pem "$@"
