import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("be.inference")
router = APIRouter()

DEFAULT_QUESTION = (
    "Identify all sound events you hear in this audio, and for each event "
    "describe its spatial location (direction and approximate distance). "
    "Report both the detected events and their localization together."
)


@router.post("/inference")
async def inference_endpoint(
    request: Request,
    model: str | None = Query(default=None, description="model id; defaults to server default"),
    question: str = Query(default=DEFAULT_QUESTION),
):
    state = request.app.state
    model_id = model or state.default_model_id

    if model_id not in state.model_status:
        log.info("inference rejected — unknown model id %r", model_id)
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model_id!r}; available: {list(state.model_status)}",
        )

    status_value = state.model_status[model_id].value
    if status_value == "loading":
        log.info("inference rejected — %s still loading", model_id)
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
        log.warning("inference rejected — %s failed: %s", model_id, err)
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
        log.warning("inference rejected — %s is None despite status=%s", model_id, status_value)
        raise HTTPException(status_code=503, detail=f"model {model_id!r} not available")

    wav_bytes = await request.body()
    if not wav_bytes:
        log.info("inference rejected — empty body")
        raise HTTPException(status_code=400, detail="empty audio body")

    log.info(
        "inference start: model=%s, bytes=%d, question=%r",
        model_id,
        len(wav_bytes),
        question,
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, instance.infer, wav_bytes, question)
    except ValueError as e:
        log.warning("inference bad input on %s: %s", model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("inference failed on %s", model_id)
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

    if isinstance(result, dict) and "model_id" not in result:
        result["model_id"] = model_id

    response_text = result.get("response", "") if isinstance(result, dict) else result
    log.info(
        "inference done: model=%s, response=%s",
        model_id,
        response_text,
    )
    return result
