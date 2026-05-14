# BE (FastAPI + audio LLM registry)

폰 앱이 보내는 stereo WAV를 받아 두 경로 중 하나로 처리한다.

| Endpoint | 입력 | 처리 |
|---|---|---|
| `POST /localize` | raw WAV bytes (`audio/wav`) | `torchaudio.load_with_torchcodec` → `processor.process_stereo()` (placeholder, GCC-PHAT 활용 가능) |
| `POST /inference?model=<id>&question=...` | raw WAV bytes (`audio/wav`) | 레지스트리에서 `<id>` 모델로 추론. 기본은 `BAT`(Spatial-AST → Q-Former → Llama-2) |
| `GET /health` | — | 전체 모델 상태/장치 노출 |

`?model`은 생략 가능 — `AUDIO_LLM_DEFAULT` 환경변수 또는 `AUDIO_LLM_ENABLED`의 첫 항목.
`?question`도 생략 시 default question 사용.

## 멀티 모델 + per-device 구성

레지스트리는 `llm.py:REGISTRY`. 새 모델을 추가하려면:
1. `models/<name>/`에 `AudioLLM` 서브클래스 작성
2. `llm.py`의 `REGISTRY`에 등록
3. `AUDIO_LLM_ENABLED`에 id 추가

환경변수:

| 변수 | 의미 | 기본값 |
|---|---|---|
| `AUDIO_LLM_ENABLED` | startup 시 로드할 모델 id (콤마 구분) | `bat` |
| `AUDIO_LLM_DEFAULT` | `/inference?model=` 미지정 시 사용할 id | enabled의 첫 항목 |
| `<MODEL_ID>_DEVICE` | 해당 모델을 올릴 torch device. 예: `BAT_DEVICE=cuda:1` | `cuda:0` (CUDA 있으면) / `cpu` |

여러 모델을 다른 GPU에 분산하려면 각각 `_DEVICE`만 다르게 주면 됨:
```bash
export AUDIO_LLM_ENABLED=bat,other
export BAT_DEVICE=cuda:0
export OTHER_DEVICE=cuda:1
```

## 모델 로딩 상태

`/health` 응답:

```json
{
  "status": "ok",
  "default_model": "bat",
  "models": {
    "bat":   {"status": "ready",   "error": null,         "device": "cuda:0"},
    "other": {"status": "loading", "error": null,         "device": null},
    "wip":   {"status": "failed",  "error": "ckpt missing", "device": null}
  }
}
```

`/inference?model=<id>` 응답:

| 모델 상태 | HTTP | 비고 |
|---|---|---|
| `loading` | **503** + `Retry-After: 30` | 폰에선 보라색 status 표시, 잠시 후 재시도 |
| `failed` | **503** + error JSON | 가중치 누락/잘못된 경로 등 |
| `ready` | 200 + 추론 결과 | 정상 |

`load_one()` 호출은 lifespan에서 백그라운드 executor로 실행돼 uvicorn은 부팅 직후 listen 시작.

## BAT 가중치 준비

이 BE는 BAT를 **vendor된 코드**로 직접 로드한다 (SLAM-LLM repo 의존 없음). 가중치 3종만 로컬에 준비.

### 1. Llama-2-7b-hf (게이트, HF token 필요)

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir /path/to/Llama-2-7b-hf
```

### 2. Spatial-AST 인코더 (~300 MB)

```bash
wget https://huggingface.co/datasets/zhisheng01/SpatialAudio/resolve/main/SpatialAST/finetuned.pth \
     -O /path/to/spatial_ast/finetuned.pth
```

### 3. BAT Q-Former + LoRA (~73 MB, 한 파일에 합쳐져있음)

```bash
wget https://huggingface.co/datasets/zhisheng01/SpatialAudio/resolve/main/BAT/model.pt \
     -O /path/to/bat/model.pt
```

### 4. `run.sh`로 기동

먼저 사용 중인 conda 환경을 활성화하고 `pip install -r requirements.txt`로 의존성을 갖춰둔 뒤:

```bash
cd localization/be
# 최초 1회 — run.sh 상단 3개 경로(BAT_LLAMA_PATH 등)를 실제 위치로 수정
chmod +x run.sh
./run.sh
```

`run.sh`가 수행하는 것:
1. 가중치 경로 3개를 사전 검증 (없으면 WARNING — 서버는 그대로 기동되어 `/health`가 `status=failed`로 노출됨)
2. 환경변수 출력 후 `uvicorn main:app --host 0.0.0.0 --port 9001` 실행

shell에서 미리 export한 변수가 스크립트 디폴트보다 우선:
```bash
BAT_DEVICE=cuda:1 BE_PORT=9002 ./run.sh
```

수동으로 가고 싶으면:
```bash
export BAT_LLAMA_PATH=... BAT_ENCODER_CKPT=... BAT_PROJECTOR_CKPT=...
uvicorn main:app --host 0.0.0.0 --port 9001
```

## 오디오 전처리 (BAT용)

BAT 인코더는 다음 입력을 요구하고, `models/bat/preprocess.py`가 자동으로 맞춘다:

| 항목 | 값 |
|---|---|
| Sample rate | **32000 Hz** |
| Channels | **2 (binaural stereo)** |
| 길이 | **10초 = 320000 샘플** |
| 정규화 | RMS → **-14 dBFS** |

- mono → 2채널 복제 (단, 진짜 binaural 정보 없음 → 성능 저하)
- 다른 sample rate → `scipy.signal.resample_poly`
- 다른 길이 → zero-pad / crop

가능하면 폰 앱 설정에서 **stereo, 32 kHz, 10초**로 보내는 게 가장 깨끗함.

## 파일 구조

```
localization/be/
├── main.py                # FastAPI app, lifespan(멀티-모델 로드), 미들웨어 로깅
├── llm.py                 # 레지스트리: enabled_model_ids / default_model_id / device_for / load_one
├── processor.py           # /localize용 process_stereo placeholder
├── tdoa.py                # GCC-PHAT 유틸
├── routers/
│   ├── localize.py        # POST /localize
│   └── inference.py       # POST /inference?model=&question= (상태 가드 + executor)
├── models/
│   ├── __init__.py
│   ├── base.py            # AudioLLM ABC (load classmethod, infer)
│   └── bat/               # vendored BAT — SLAM-LLM repo 불필요
│       ├── __init__.py
│       ├── model.py       # BAT(AudioLLM) — load/infer 오케스트레이션
│       ├── spatial_ast.py # vendored BinauralEncoder
│       ├── projector.py   # vendored EncoderProjectorQFormer
│       ├── preprocess.py  # waveform 전처리 + format_prompt
│       └── ATTRIBUTION.md # SLAM-LLM Apache-2.0 attribution
└── requirements.txt
```

## 로그 포맷

매 요청마다 미들웨어가 한 줄 access log:
```
2026-05-13 21:00:12 INFO be: POST /inference → 200 (3421.5ms, client=192.168.0.10)
```

라우터 내부 INFO:
```
INFO be.localize: localize: shape=(2, 480000), sr=48000 Hz, bytes=1920044
INFO be.inference: inference start: model=bat, bytes=320044, question='What can you hear...'
INFO be.inference: inference done: model=bat, response_len=128
```

## 다른 audio LLM 추가하기

1. `models/<id>/` 디렉터리 생성, `AudioLLM` 서브클래스 작성:
   ```python
   class OtherModel(AudioLLM):
       model_id = "other"

       @classmethod
       def load(cls, device): ...

       def infer(self, wav_bytes, question): ...
   ```
2. `llm.py`의 `REGISTRY`에 등록:
   ```python
   from models.other import OtherModel
   REGISTRY = {BAT.model_id: BAT, OtherModel.model_id: OtherModel}
   ```
3. 실행 시:
   ```bash
   AUDIO_LLM_ENABLED=bat,other BAT_DEVICE=cuda:0 OTHER_DEVICE=cuda:1 uvicorn ...
   ```

폰 앱에서는 `/inference?model=other`로 라우팅.

## 비범위

- HF token / Llama-2 게이트 액세스 (사용자 사전 처리)
- `processor.py` 실제 azimuth 계산 (별도 swap-in)
- GPU OOM 자동 복구 — load 실패는 `model_status=failed`로 격리, 재시작 필요
