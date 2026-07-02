# Physics Impact Simulator — BE

3D 메쉬 + action을 받아 **교체 가능한 mesh-to-mesh 모델**로 변형 결과를 돌려주는
FastAPI 백엔드. 구조는 `seld/be` · `localization/be`를 미러링한다. 등록 모델:
`free_fall`(데모 기본: 자유 낙하 다중 접촉) · `metal_dent`(충격: 시간-시퀀스 dent) ·
`dummy`(단일 프레임 선형 감쇠).

## 핵심: Pluggable Model Interface

```
models/base.py       BaseMeshPredictor (ABC) — load() / predict() / simulate()
models/free_fall/    FreeFallSimulator     (자유 낙하 다중 접촉, 기본 모델)
models/metal_dent/   MetalDentSimulator    (충격 시간-시퀀스 dent)
models/dummy/        DummyLinearDeformer   (단일 프레임 예시)
config.py            REGISTRY + 환경변수 기반 동적 로드 (싱글톤)
```
`free_fall`·`metal_dent`의 궤적 수식은 FE 오프라인 미러(fe/free_fall_sim.py,
fe/app.py metal_dent_simulate)와 바이트/수식 동일 — 한쪽만 고치면 안 된다.

새 모델 도입 절차 (main.py · routers 무수정):
1. `models/<id>/model.py`에 `BaseMeshPredictor` 상속 클래스 작성 (`model_id`, `load`, `predict`).
2. `config.py`에 import 1줄 + `REGISTRY` 항목 1줄 추가.
3. `MESH_MODEL_ENABLED=<id>`로 활성화.

## 구조

```
be/
├── main.py              # FastAPI: lifespan 싱글톤 로드, /health, /predict 마운트
├── config.py            # 모델 레지스트리 + 동적 로드 설정
├── models/
│   ├── base.py          # BaseMeshPredictor (인터페이스)
│   └── dummy/model.py   # DummyLinearDeformer
├── utils/mesh_handler.py# meshio load/write (Base64↔vertices/faces)
├── utils/chat_fallback.py# /chat rule-based 폴백 (FE fe/chat_fallback.py와 미러)
├── routers/predict.py   # POST /predict + Pydantic 스키마
├── routers/chat.py      # POST /chat — LLM(OpenAI-호환) QA + in-band 폴백
├── smoke_test.py        # end-to-end 검증 스크립트
├── requirements.txt
└── run.sh               # 0.0.0.0:9003
```

## 실행

```bash
pip install -r requirements.txt
./run.sh                 # 0.0.0.0:9003 (localization 9001 / seld 9002와 동시 구동 가능)
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `MESH_MODEL_ENABLED` | `free_fall,metal_dent,dummy` | 로드할 모델 id (콤마 구분) |
| `MESH_MODEL_DEFAULT` | `free_fall` | `/predict?model=` 미지정 시 기본 (= 첫 enabled) |
| `<MODEL_ID>_DEVICE` | `cpu` | 모델별 장치 (예: `DUMMY_DEVICE=cpu`) |
| `BE_HOST` / `BE_PORT` | `0.0.0.0` / `9003` | 서버 바인딩 |
| `CHAT_LLM_BASE_URL` | (없음) | `/chat`용 OpenAI-호환 API base (예: `https://api.openai.com/v1`, `http://127.0.0.1:11434/v1`) |
| `CHAT_LLM_MODEL` | (없음) | 모델명 (예: `gpt-4o-mini`, `llama3`) — BASE_URL과 둘 다 있어야 LLM 모드 |
| `CHAT_LLM_API_KEY` | (없음) | Bearer 키 (로컬 서버는 생략 가능) |
| `CHAT_LLM_TIMEOUT` | `30` | LLM 호출 타임아웃(초) |

## API

### `GET /health`

서버 연결 + 모델 로드 상태. 모든 모델 READY면 **200**, 하나라도
LOADING/FAILED면 **503** (FE 상태 점: 초록/보라).

```json
{"status": "ok", "default_model": "dummy", "uptime_s": 1.2,
 "models": {"dummy": {"status": "ready", "error": null, "device": "cpu"}}}
```

### `POST /predict?model=<id>`

`?model=` 미지정 시 `MESH_MODEL_DEFAULT` 사용.

요청 (JSON):
```json
{
  "mesh_base64": "<base64 mesh file bytes>",
  "file_format": "vtk",
  "action": {"impact_node": 102, "force": [0.0, -10.0, 0.0]}
}
```
`file_format`은 파일 확장자 — **meshio가 지원하는 모든 포맷**을 받는다
(`vtk`/`vtu`/`obj`/`stl`/`ply`/`off`/`msh`/`bdf`/`inp`/`mesh`/`med`/… meshio가
확장자로 자동 추론). triangle/quad 표면이 있으면 그 표면을, 체적 메쉬
(tetra/hexahedron/wedge/pyramid)면 **경계면(boundary surface)**을 자동 추출해 변형한다.
`action`은 모델로 그대로 전달된다. dummy가 인식하는 키:
`impact_node`(int), `force`([x,y,z]), 선택 `radius`(감쇠 반경), `scale`(배율).

응답 (JSON):
```json
{"success": true, "result_mesh_base64": "<base64>", "model_id": "dummy", "num_vertices": 121}
```

에러: 잘못된 base64/메쉬/action → **400**, 모델 미준비 → **503**, 모델 내부 오류 → **500**.

### `POST /simulate?model=<id>`

충격에 대한 **시간에 따른 변형 궤적**(애니메이션용 프레임 시퀀스)을 반환한다. 요청
스키마는 `/predict`와 동일하며 `action`에 프레임 수 `frames`(기본 60)를 넣을 수 있다.
프레임은 topology(faces)를 공유하므로 메쉬를 T번 직렬화하지 않고 **바이너리 페이로드**로
보낸다:

```json
{ "success": true, "model_id": "metal_dent", "num_frames": 60, "num_vertices": 1681,
  "num_faces": 3200,
  "faces_b64":  "<base64 int32 (M,3) C-order>",
  "frames_b64": "<base64 float32 (T,N,3) C-order>" }
```

action 키는 모델별로 다르다:
- **free_fall** (기본): `drop_height`(>0, 기본 1.0) · `restitution`([0,1), 기본 0.3) ·
  `orientation`([rx,ry,rz]°, 낙하 자세) · `scale` · `frames`(기본 90) · 선택 `radius`.
  낙하 자세 기준 최저 밴드를 클러스터링해 **여러 접촉점에 dent**를 주고 반발계수로
  감쇠 바운스한다. `frames[0]`=공중, `frames[-1]`=바닥 정착.
- **metal_dent** (충격): `impact_node` · `force`([x,y,z]) · `scale` · 선택 `radius` · `frames`.

기본 모델 `free_fall`(FreeFallSimulator)은 절차적 자유 낙하 데모. `metal_dent`는 영구
소성 dent + 감쇠 링잉. 단일 프레임 모델(dummy 등)은 `BaseMeshPredictor.simulate`
기본 구현이 predict를 `(1,N,3)`로 감싸 그대로 동작한다.

> 모델 교체: `models/<id>/`에 `BaseMeshPredictor` 상속(시퀀스면 `simulate` 오버라이드) +
> `config.py` 등록만으로 `/predict`·`/simulate` 둘 다 새 모델로 서빙된다.

### `POST /chat`

시뮬레이션 **분석 결과에 대한 QA**. FE가 만든 압축 분석 요약(JSON)과 질문을 받아,
`CHAT_LLM_*` env가 설정돼 있으면 **OpenAI-호환** `chat/completions`로 답하고
(`mode: "llm"`), 미설정/호출 실패 시 rule-based 답변으로 강등한다(`mode: "fallback"`,
`error`에 사유). **LLM 실패는 5xx가 아니라 in-band 폴백** — 데모가 외부 API 상태에
좌우되지 않는다. mesh 모델 레지스트리와 무관해 모델 로딩 중에도 동작한다.

요청/응답 (JSON):
```json
{"question": "태양전지판 괜찮아?", "analysis": {"components": [...]},
 "history": [{"role": "user", "content": "..."}]}
```
```json
{"success": true, "answer": "...", "mode": "llm", "model": "gpt-4o-mini", "error": null}
```

rule-based 엔진(`utils/chat_fallback.py`)은 FE `fe/chat_fallback.py`와 **바이트 단위
미러** — 서버 미연결 시 FE가 로컬에서 같은 답을 낸다. 한쪽만 수정 금지.
`/health` 응답의 `chat.llm_configured`로 현재 모드를 확인할 수 있다.

## 검증

서버를 띄운 뒤:
```bash
python smoke_test.py     # 샘플 그리드 → /chat 폴백 + /predict 변형 검증 (OK ✓)
```
