# Physics Impact Simulator — BE

3D 메쉬 + 충격(action)을 받아 **교체 가능한 mesh-to-mesh 모델**로 변형 결과를
돌려주는 FastAPI 백엔드. 구조는 `seld/be` · `localization/be`를 미러링한다. 등록 모델:
`metal_dent`(데모: 시간-시퀀스 dent, 기본값) · `dummy`(단일 프레임 선형 감쇠).

## 핵심: Pluggable Model Interface

```
models/base.py       BaseMeshPredictor (ABC) — load() / predict() / simulate()
models/dummy/        DummyLinearDeformer   (단일 프레임 예시)
models/metal_dent/   MetalDentSimulator    (시간-시퀀스 데모)
config.py            REGISTRY + 환경변수 기반 동적 로드 (싱글톤)
```

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
├── routers/predict.py   # POST /predict + Pydantic 스키마
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
| `MESH_MODEL_ENABLED` | `dummy` | 로드할 모델 id (콤마 구분) |
| `MESH_MODEL_DEFAULT` | `dummy` | `/predict?model=` 미지정 시 기본 |
| `<MODEL_ID>_DEVICE` | `cpu` | 모델별 장치 (예: `DUMMY_DEVICE=cpu`) |
| `BE_HOST` / `BE_PORT` | `0.0.0.0` / `9003` | 서버 바인딩 |

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

기본 모델 `metal_dent`(MetalDentSimulator)는 절차적 금속 dent 데모 — 영구 소성 dent +
감쇠 링잉. 단일 프레임 모델(dummy 등)은 `BaseMeshPredictor.simulate` 기본 구현이
predict를 `(1,N,3)`로 감싸 그대로 동작한다.

> 모델 교체: `models/<id>/`에 `BaseMeshPredictor` 상속(시퀀스면 `simulate` 오버라이드) +
> `config.py` 등록만으로 `/predict`·`/simulate` 둘 다 새 모델로 서빙된다.

## 검증

서버를 띄운 뒤:
```bash
python smoke_test.py     # 샘플 그리드 → /predict → 변형 검증 (OK ✓)
```
