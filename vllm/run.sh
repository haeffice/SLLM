#!/usr/bin/env bash
# Start the CPU vLLM FastAPI backend.
#
# Activate your Python environment first, then:
#   chmod +x run.sh   # once
#   ./run.sh
#
# Each export uses `${VAR:-default}` so values already set in the shell
# take precedence over the defaults in this file.
#
# IMPORTANT: VLLM_CPU_* and LD_PRELOAD MUST be exported here (before
# `exec uvicorn`) — the vLLM C++/OpenMP runtime reads them at process
# init and they CANNOT be set later from inside the app. Run
# `python preflight_cli.py "$VLLM_BE_MODEL_PATH"` first and copy the
# recommended values below.

set -euo pipefail

# =============================================================================
# 1) 모델 경로 — 호스트가 직접 HF에서 다운로드한 로컬 디렉터리
# =============================================================================
export VLLM_BE_MODEL_PATH="${VLLM_BE_MODEL_PATH:-/path/to/hf-model-dir}"

# =============================================================================
# 2) 엔진 설정 (선택)
# =============================================================================
export VLLM_BE_DTYPE="${VLLM_BE_DTYPE:-bfloat16}"          # CPU는 bfloat16 권장
export VLLM_BE_TP_SIZE="${VLLM_BE_TP_SIZE:-}"              # 공백=auto(NUMA 노드 수)
export VLLM_BE_MAX_NUM_SEQS="${VLLM_BE_MAX_NUM_SEQS:-256}"
export VLLM_BE_MAX_NUM_BATCHED_TOKENS="${VLLM_BE_MAX_NUM_BATCHED_TOKENS:-4096}"
export VLLM_BE_BLOCK_SIZE="${VLLM_BE_BLOCK_SIZE:-128}"     # 32의 배수
export VLLM_BE_MAX_MODEL_LEN="${VLLM_BE_MAX_MODEL_LEN:-}"  # 공백=config.json
export VLLM_BE_QUANTIZATION="${VLLM_BE_QUANTIZATION:-}"    # awq|gptq|compressed-tensors
export VLLM_BE_PREFLIGHT_MODE="${VLLM_BE_PREFLIGHT_MODE:-warn}"  # warn|enforce|off

# =============================================================================
# 3) 프로세스 레벨 vLLM 자원 설정 — preflight 권장값으로 채울 것
#     (`python preflight_cli.py "$VLLM_BE_MODEL_PATH"` 출력 참고)
# =============================================================================
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-40}"          # GiB
export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-auto}"  # 예: 0-31|32-63
export VLLM_CPU_NUM_OF_RESERVED_CPU="${VLLM_CPU_NUM_OF_RESERVED_CPU:-1}"

# tcmalloc + Intel OpenMP preload (없으면 빈 값으로 두고 설치 권장)
_TC="$(ldconfig -p 2>/dev/null | awk -F'=> ' '/libtcmalloc_minimal.so.4/{print $2; exit}')"
_IOMP="$(ldconfig -p 2>/dev/null | awk -F'=> ' '/libiomp5.so/{print $2; exit}')"
export LD_PRELOAD="${LD_PRELOAD:-${_TC}${_IOMP:+:${_IOMP}}}"

# =============================================================================
# 4) 서버 바인딩
# =============================================================================
HOST="${BE_HOST:-0.0.0.0}"
PORT="${BE_PORT:-9001}"

# =============================================================================
# 5) (이하 수정 불필요)
# =============================================================================
BE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BE_DIR"

echo "==> model: $VLLM_BE_MODEL_PATH"
if [[ ! -d "$VLLM_BE_MODEL_PATH" ]]; then
    echo "    [MISSING] 모델 디렉터리가 없습니다 — 엔진 로드가 실패하고"
    echo "              /health 의 model_status=failed 로 노출됩니다."
fi
echo "==> LD_PRELOAD=${LD_PRELOAD:-(none)}"
echo "==> VLLM_CPU_KVCACHE_SPACE=$VLLM_CPU_KVCACHE_SPACE  "\
"VLLM_CPU_OMP_THREADS_BIND=$VLLM_CPU_OMP_THREADS_BIND"
echo "==> starting uvicorn on ${HOST}:${PORT}"

exec uvicorn main:app --host "$HOST" --port "$PORT" "$@"
