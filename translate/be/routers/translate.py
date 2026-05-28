import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("be.translate")
router = APIRouter()


@router.post("/translate")
async def translate_endpoint(
    request: Request,
    model: str | None = Query(default=None, description="model id; defaults to server default"),
    src: str = Query(default="en", description="source language code (e.g. en, ko)"),
    tgt: str = Query(default="ko", description="target language code (e.g. ko, en)"),
):
    state = request.app.state
    model_id = model or state.default_model_id

    if model_id not in state.model_status:
        log.info("translate rejected — unknown model id %r", model_id)
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model_id!r}; available: {list(state.model_status)}",
        )

    status_value = state.model_status[model_id].value
    if status_value == "loading":
        log.info("translate rejected — %s still loading", model_id)
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
        log.warning("translate rejected — %s failed: %s", model_id, err)
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
        log.warning("translate rejected — %s is None despite status=%s", model_id, status_value)
        raise HTTPException(status_code=503, detail=f"model {model_id!r} not available")

    wav_bytes = await request.body()
    if not wav_bytes:
        log.info("translate rejected — empty body")
        raise HTTPException(status_code=400, detail="empty audio body")

    log.info(
        "translate start: model=%s, bytes=%d, src=%s, tgt=%s",
        model_id,
        len(wav_bytes),
        src,
        tgt,
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, instance.translate, wav_bytes, src, tgt)
    except ValueError as e:
        log.warning("translate bad input on %s: %s", model_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("translate failed on %s", model_id)
        raise HTTPException(status_code=500, detail=f"translate error: {e}")

    if isinstance(result, dict) and "model_id" not in result:
        result["model_id"] = model_id

    log.info(
        "translate done: model=%s, text=%s",
        model_id,
        result.get("text", "") if isinstance(result, dict) else "",
    )
    return result
