#!/usr/bin/env bash
# Start the Physics Impact Simulator BE FastAPI server (dummy deformer).
#
# Activate your Python environment first, then:
#   chmod +x run.sh   # once
#   ./run.sh
#
# Each export uses `${VAR:-default}` so values already set in the shell
# take precedence over the defaults in this file.

set -euo pipefail

# =============================================================================
# 1) 실제 mesh-to-mesh 모델 가중치 경로 — 추후 기재
#    (models/<id>/ 추가 + config.py REGISTRY 한 줄 + MESH_MODEL_ENABLED 변경)
# =============================================================================
# export MESH_CKPT="${MESH_CKPT:-/path/to/mesh/model.pt}"

# =============================================================================
# 2) (선택) 모델 / 장치
# =============================================================================
export DUMMY_DEVICE="${DUMMY_DEVICE:-cpu}"                # dummy는 cpu로 충분
export MESH_MODEL_ENABLED="${MESH_MODEL_ENABLED:-dummy}"  # 콤마 구분으로 추가 모델 등록
export MESH_MODEL_DEFAULT="${MESH_MODEL_DEFAULT:-dummy}"  # /predict?model= 미지정 시 기본

# =============================================================================
# 3) 서버 바인딩 — localization(9001)/seld(9002)와 동시 구동 가능하도록 9003
# =============================================================================
HOST="${BE_HOST:-0.0.0.0}"
PORT="${BE_PORT:-9003}"

# =============================================================================
# 4) (이하 수정 불필요)
# =============================================================================
BE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BE_DIR"

echo "==> MESH_MODEL_ENABLED=$MESH_MODEL_ENABLED  MESH_MODEL_DEFAULT=$MESH_MODEL_DEFAULT  DUMMY_DEVICE=$DUMMY_DEVICE"
echo "==> starting uvicorn on ${HOST}:${PORT}"
exec uvicorn main:app --host "$HOST" --port "$PORT" "$@"
