"""POST /simulate — 메쉬 + 충격(action) → 시간에 따른 변형 프레임 시퀀스.

/predict가 단일 변형 메쉬를 주는 반면 /simulate는 (T,N,3) 궤적을 준다. 프레임은
topology(faces)를 공유하므로 메쉬를 T번 직렬화하지 않고 효율적 바이너리 페이로드로
보낸다: faces(little-endian int32)·frames(little-endian float32)를 각각 base64로
1번씩. FE가 np.frombuffer(<i4/<f4)로 복원. (같은 아키텍처 가정 제거 — 엔디안 고정.)

요청 스키마는 /predict와 동일(PredictRequest). action에 프레임 수 `frames`를 넣을 수 있다.
모델 상태 게이팅은 routers/predict.py 규약을 그대로 따른다.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from models.base import BaseMeshPredictor
from routers.predict import PredictRequest
from utils.mesh_handler import load_mesh_from_bytes

log = logging.getLogger("be.simulate")
router = APIRouter()

# T*N 상한 — /simulate OOM 방지(데모 안전장치). 초과 시 400.
MAX_SIM_POINTS = 15_000_000


def _run_simulate(
    instance: BaseMeshPredictor, raw: bytes, file_format: str, action: dict
) -> tuple[np.ndarray, np.ndarray]:
    """동기 파이프라인: 파싱 → 크기검사 → simulate → 검증 → (faces '<i4', frames '<f4').

    입력 ValueError(파싱·잘못된 action)는 400, 모델 출력 계약 위반(비배열·형상)은
    RuntimeError로 바꿔 500. 바이너리 페이로드는 little-endian(int32/float32) 고정.
    """
    vertices, faces = load_mesh_from_bytes(raw, file_format)  # ValueError → 400 (입력)

    # OOM 방지: 정점수 × 프레임수(최악치) 상한. 초과 시 400.
    try:
        frames_hint = min(max(int(action.get("frames", 60)), 2), 240)
    except (TypeError, ValueError):
        frames_hint = 240
    n = int(vertices.shape[0])
    if n * frames_hint > MAX_SIM_POINTS:
        raise ValueError(
            f"mesh too large for /simulate: {n} vertices × {frames_hint} frames "
            f"exceeds {MAX_SIM_POINTS} points"
        )

    raw_frames = instance.simulate(vertices, faces, action)  # ValueError → 400 (잘못된 action)
    try:
        frames = np.asarray(raw_frames, dtype=np.float64)
    except (ValueError, TypeError) as e:  # 모델이 배열이 아닌 출력 → 서버측 500
        raise RuntimeError(f"model returned non-array frames: {e}") from e
    if frames.ndim != 3 or frames.shape[0] < 1 or frames.shape[1:] != vertices.shape:
        raise RuntimeError(
            f"model returned frames {frames.shape}, expected (T>=1,{n},3)"
        )
    faces32 = np.ascontiguousarray(faces, dtype="<i4")
    frames32 = np.ascontiguousarray(frames, dtype="<f4")
    return faces32, frames32


@router.post("/simulate")
async def simulate_endpoint(
    req: PredictRequest,
    request: Request,
    model: str | None = Query(default=None, description="model id; defaults to server default"),
):
    state = request.app.state
    model_id = model or state.default_model_id

    # --- 모델 상태 게이팅 (routers/predict.py와 동일) -----------------------
    if model_id not in state.model_status:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model_id!r}; available: {list(state.model_status)}",
        )
    status_value = state.model_status[model_id].value
    if status_value == "loading":
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "30"},
            content={"detail": f"model {model_id!r} is still loading, please retry shortly",
                     "model_id": model_id, "model_status": "loading"},
        )
    if status_value == "failed":
        err = state.model_errors.get(model_id)
        return JSONResponse(
            status_code=503,
            content={"detail": f"model {model_id!r} failed to load at startup",
                     "model_id": model_id, "model_status": "failed", "error": err},
        )
    instance = state.models.get(model_id)
    if instance is None:
        raise HTTPException(status_code=503, detail=f"model {model_id!r} not available")

    # --- Base64 디코드 -------------------------------------------------------
    try:
        raw = base64.b64decode(req.mesh_base64, validate=True)  # binascii.Error ⊂ ValueError
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid mesh_base64: {e}")
    if not raw:
        raise HTTPException(status_code=400, detail="empty mesh data")

    log.info(
        "simulate start: model=%s, format=%s, bytes=%d, action=%s",
        model_id, req.file_format, len(raw), req.action,
    )

    # --- 시뮬레이션 (CPU-bound → executor) ----------------------------------
    try:
        loop = asyncio.get_running_loop()
        faces32, frames32 = await loop.run_in_executor(
            None, _run_simulate, instance, raw, req.file_format, req.action
        )
    except ValueError as e:
        log.warning("simulate bad input on %s: %s", model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("simulate failed on %s", model_id)
        raise HTTPException(status_code=500, detail=f"simulation error: {e}")

    num_frames, num_vertices, _ = frames32.shape
    log.info(
        "simulate done: model=%s, T=%d, N=%d, faces=%d",
        model_id, num_frames, num_vertices, faces32.shape[0],
    )
    return {
        "success": True,
        "model_id": model_id,
        "num_frames": int(num_frames),
        "num_vertices": int(num_vertices),
        "num_faces": int(faces32.shape[0]),
        "faces_b64": base64.b64encode(faces32.tobytes()).decode("ascii"),
        "frames_b64": base64.b64encode(frames32.tobytes()).decode("ascii"),
    }
