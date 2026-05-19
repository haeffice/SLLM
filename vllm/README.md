# CPU vLLM Serving

CPU 전용 환경에서 대형 로컬 HuggingFace 모델을 vLLM으로 서빙하고, FastAPI
백엔드로 노출한다. `AsyncLLMEngine` 은 서버 시작 시 **한 번만** 로드되어
상주하며, 모든 요청을 continuous batching 으로 처리한다.

## 구성

| 파일 | 역할 |
|---|---|
| `config.py` | env 기반 설정 (`VLLM_BE_*`) |
| `cpu_topology.py` | 호스트 RAM / NUMA / ISA 탐지 (vllm·torch 불필요) |
| `preflight.py` | 동적 권장 사양 체커 (vllm·torch 불필요) |
| `preflight_cli.py` | `python preflight_cli.py <model_dir>` 단독 실행 |
| `engine.py` | `AsyncLLMEngine` 빌더 (vllm 지연 import) |
| `main.py` | FastAPI 앱, lifespan 로드, `/health`, `/preflight` |
| `routers/generate.py` | `POST /generate`, `POST /generate/stream` |
| `runner.py` | 클라이언트: SSH 포트포워딩 + 요청 |
| `run.sh` | env export 후 `exec uvicorn main:app` |

## 개발 PC vs 실제 배포 서버

prebuilt vLLM CPU wheel 은 **AVX512** 로 컴파일되어 있어 실제 배포 대상인
**AMX Xeon** 서버에서 동작한다. AVX2 전용 개발 PC 에서는 wheel 자체가
import/실행되지 않으므로 — 그곳에서는 코드 작성과 `preflight_cli.py`
실행만 가능하다(ISA `FAIL` 이 정상이며 예상된 결과). 개발 PC 에서도 서버는
정상 기동된다: 엔진 로드만 깔끔히 실패하고 `model_status=failed` 로
표시되며 `/health`·`/preflight` 는 계속 동작한다.

## 0단계: 권장 사양 체크

```bash
python preflight_cli.py /path/to/hf-model-dir
```

모델 파일로부터 가중치 + KV 캐시 + per-rank 메모리를 추정하고, 실행
호스트(RAM, NUMA 노드별 RAM, ISA, Python/gcc)를 점검한다. 항목별로
`PASS`/`WARN`/`FAIL` 과 함께 권장 env 값(`VLLM_BE_TP_SIZE`,
`VLLM_CPU_KVCACHE_SPACE`, `VLLM_CPU_OMP_THREADS_BIND`,
`VLLM_CPU_NUM_OF_RESERVED_CPU`, `LD_PRELOAD`)을 출력한다.
가장 중요한 검사는 **per-rank ≤ 최소 NUMA 노드 RAM** 으로, 이를 초과하면
TP 워커가 OOM 으로 강제 종료된다(exitcode 9).

종료 코드: `0`=PASS, `1`=WARN, `2`=FAIL.

## 서버 실행

1. `python preflight_cli.py "$VLLM_BE_MODEL_PATH"` 실행 후, 출력된 권장
   env 값을 `run.sh` 3번 섹션(`VLLM_CPU_*`, `LD_PRELOAD`)에 반영한다.
   이 값들은 **반드시 uvicorn 시작 전에** 설정되어야 한다(vLLM C++/OpenMP
   런타임이 프로세스 init 때 읽으므로 앱 내부에서는 설정 불가).
2. `chmod +x run.sh && ./run.sh`
3. 로그에서 preflight → 백그라운드 엔진 로드 → `engine ready` 순으로 확인.
   `GET /health` 가 `model_status: ready` 가 되면 준비 완료.

`VLLM_BE_PREFLIGHT_MODE=enforce` 로 두면 `FAIL` 시 서버 기동을 중단한다
(기본값 `warn` 은 로그만 남기고 계속 진행).

## 클라이언트 (runner.py)

```bash
# SSH 포워딩 자동 설정 후 localhost 로 통신 (이미 포워딩돼 있으면 재사용)
python runner.py --ssh-target user@xeon --remote-port 9001 \
                 --local-port 9001 --prompt "안녕" --stream

# 같은 호스트 / 이미 포워딩된 경우: SSH 설정 건너뜀
python runner.py --no-forward --remote-host 127.0.0.1 \
                 --remote-port 9001 --prompt "안녕"
```

로컬 포트에 호환 서버가 이미 응답하면 SSH 단계를 건너뛰고 해당 터널을
재사용하며, 종료 시에도(직접 만든 터널이 아니므로) 그대로 둔다.
`--wait N` 은 모델이 ready 가 될 때까지 `/health` 를 최대 N초 폴링한다.
`--prompt -` 는 표준입력을 프롬프트로 사용한다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 모델 상태 / 에러 / preflight 요약 / 엔진 정보 |
| GET | `/preflight` | 실행 호스트 기준 권장 사양 재검사 (JSON) |
| POST | `/generate` | 단일 JSON 응답 |
| POST | `/generate/stream` | SSE 토큰 델타 스트리밍 |

`/generate` 요청 본문 예시:

```json
{
  "prompt": "...",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": -1,
  "stop": ["</s>"],
  "seed": 42,
  "stream": false
}
```

모델 로딩 중에는 503(`Retry-After: 30`), 로드 실패 시에는 503 +
에러 메시지를 반환한다(`localization/be` 가드 패턴과 동일).

## 검증 절차

### 개발 PC (AVX2, 모델 없음)
1. `python preflight_cli.py <임의 HF dir>` → ISA `FAIL`(avx512f 없음)
   예상, 표 렌더링 및 numactl 파싱 확인.
2. 작은 모델(`facebook/opt-125m` 로컬 다운로드)로 preflight 추정 확인.
3. `VLLM_BE_MODEL_PATH=/tmp/opt125m VLLM_BE_PREFLIGHT_MODE=warn ./run.sh`
   → 기동, preflight 비치명, `/health`=failed(또는 wheel import 시 로드).
   `GET /health`, `GET /preflight` 확인.
4. `python runner.py --no-forward --remote-host 127.0.0.1
   --remote-port 9001 --prompt "Hello" --stream` → SSH 없이 SSE 경로 검증.

### 실제 Xeon 서버 (200B, 로컬 경로)
1. CPU wheel 설치:
   ```bash
   export VLLM_VERSION=$(curl -s \
     https://api.github.com/repos/vllm-project/vllm/releases/latest \
     | jq -r .tag_name | sed 's/^v//')
   uv pip install \
     https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl \
     --torch-backend cpu
   ```
2. `python preflight_cli.py /data/models/<200b>` → `PASS`(또는 RAM tight
   `WARN`). 권장 `tp_size`(=NUMA 노드 수), `VLLM_CPU_KVCACHE_SPACE`,
   `VLLM_CPU_OMP_THREADS_BIND` 확인.
3. 권장 env 를 `run.sh` 에 반영, `VLLM_BE_PREFLIGHT_MODE=enforce`,
   `./run.sh` → preflight PASS, 백그라운드 로드 elapsed, `READY` 로그,
   worker exitcode 9 미발생 확인.
4. 개발 PC 에서 `python runner.py --ssh-target user@xeon
   --remote-port 9001 --local-port 9001 --prompt "..." --stream`
   → `-N -L` 자동 포워딩, `/health` 폴링 후 토큰 스트림. 재실행 시
   터널 재사용. 스트림 중 Ctrl-C → 서버가 `engine.abort` 로 슬롯 해제.
5. runner 동시 다중 실행 → continuous batching·`max_num_seqs` 확인.
