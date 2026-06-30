"""POST /predict — 메쉬 + 충격(action) → 변형된 메쉬.

흐름: Base64 디코드 → meshio 파싱(vertices/faces) → 활성 모델 predict →
new_vertices + faces로 메쉬 재빌드 → Base64 인코딩. CPU-bound 구간
(meshio + predict)은 run_in_executor로 이벤트 루프 밖에서 처리한다.

모델 상태 게이팅(unknown→400, loading/failed→503)은 다른 BE(seld/localization)의
/inference 라우터 규약을 그대로 따른다.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models.base import BaseMeshPredictor
from utils.mesh_handler import load_mesh_from_bytes, write_mesh_to_bytes

log = logging.getLogger("be.predict")
router = APIRouter()


class PredictRequest(BaseModel):
    mesh_base64: str = Field(..., description="Base64-encoded mesh file bytes")
    file_format: str = Field("vtk", description="mesh file format, e.g. 'vtk', 'obj'")
    action: dict = Field(
        ...,
        description='impact spec, e.g. {"impact_node": 102, "force": [0.0, -10.0, 0.0]}',
    )


class PredictResponse(BaseModel):
    success: bool
    result_mesh_base64: str | None = None
    model_id: str | None = None
    num_vertices: int | None = None
    error: str | None = None


def _deform(
    instance: BaseMeshPredictor, raw: bytes, file_format: str, action: dict
) -> tuple[bytes, int]:
    """동기 파이프라인: 파싱 → predict → 검증 → 직렬화. executor에서 호출.

    입력 단계 ValueError(메쉬 파싱·잘못된 action)는 호출부에서 400으로,
    그 외(모델 출력 형식 위반·직렬화 실패 등 서버측 문제)는 RuntimeError로
    바꿔 500으로 매핑되게 한다 — 클라이언트 잘못이 아닌 오류를 400으로
    오인하지 않도록.
    """
    vertices, faces = load_mesh_from_bytes(raw, file_format)  # ValueError → 400 (입력)
    raw_out = instance.predict(vertices, faces, action)  # ValueError → 400 (잘못된 action)
    try:
        new_vertices = np.asarray(raw_out, dtype=np.float64)
    except (ValueError, TypeError) as e:  # 모델 출력이 (N,3) 배열이 아님 → 서버측
        raise RuntimeError(f"model returned non-array output: {e}") from e
    if new_vertices.shape != vertices.shape:
        # 정점 개수/형상이 바뀌면 faces와 결합할 수 없다 → 서버측 계약 위반.
        raise RuntimeError(
            f"model returned vertices {new_vertices.shape}, expected {vertices.shape}"
        )
    try:
        out_bytes = write_mesh_to_bytes(new_vertices, faces, file_format)
    except ValueError as e:  # 직렬화 실패 → 서버측
        raise RuntimeError(f"failed to serialize result mesh: {e}") from e
    return out_bytes, int(new_vertices.shape[0])


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    req: PredictRequest,
    request: Request,
    model: str | None = Query(default=None, description="model id; defaults to server default"),
):
    state = request.app.state
    model_id = model or state.default_model_id

    # --- 모델 상태 게이팅 (seld/localization 규약과 동일) ---------------------
    if model_id not in state.model_status:
        log.info("predict rejected — unknown model id %r", model_id)
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model_id!r}; available: {list(state.model_status)}",
        )

    status_value = state.model_status[model_id].value
    if status_value == "loading":
        log.info("predict rejected — %s still loading", model_id)
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "30"},
            content={
                "detail": f"model {model_id!r} is still loading, please retry shortly",
                "model_id": model_id,
                "model_status": "loading",
            },
        )
    if status_value == "failed":
        err = state.model_errors.get(model_id)
        log.warning("predict rejected — %s failed: %s", model_id, err)
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"model {model_id!r} failed to load at startup",
                "model_id": model_id,
                "model_status": "failed",
                "error": err,
            },
        )

    instance = state.models.get(model_id)
    if instance is None:
        log.warning("predict rejected — %s is None despite status=%s", model_id, status_value)
        raise HTTPException(status_code=503, detail=f"model {model_id!r} not available")

    # --- Base64 디코드 -------------------------------------------------------
    try:
        # binascii.Error는 ValueError 하위 클래스 → ValueError 하나로 충분.
        raw = base64.b64decode(req.mesh_base64, validate=True)
    except ValueError as e:
        log.info("predict rejected — bad base64: %s", e)
        raise HTTPException(status_code=400, detail=f"invalid mesh_base64: {e}")
    if not raw:
        raise HTTPException(status_code=400, detail="empty mesh data")

    log.info(
        "predict start: model=%s, format=%s, bytes=%d, action=%s",
        model_id,
        req.file_format,
        len(raw),
        req.action,
    )

    # --- 변형 (CPU-bound → executor) ----------------------------------------
    try:
        loop = asyncio.get_running_loop()
        out_bytes, num_vertices = await loop.run_in_executor(
            None, _deform, instance, raw, req.file_format, req.action
        )
    except ValueError as e:  # 잘못된 메쉬/action
        log.warning("predict bad input on %s: %s", model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("predict failed on %s", model_id)
        raise HTTPException(status_code=500, detail=f"prediction error: {e}")

    result_b64 = base64.b64encode(out_bytes).decode("ascii")
    log.info("predict done: model=%s, N=%d, out_bytes=%d", model_id, num_vertices, len(out_bytes))
    return PredictResponse(
        success=True,
        result_mesh_base64=result_b64,
        model_id=model_id,
        num_vertices=num_vertices,
    )
