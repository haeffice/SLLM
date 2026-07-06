"""Text generation endpoints.

POST /generate         — single JSON response
POST /generate/stream  — Server-Sent Events (token deltas)

Guard pattern (503 while LOADING / FAILED, 503 if engine missing) mirrors
``localization/be/routers/inference.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger("be.generate")
router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False


def _guard(state) -> JSONResponse | None:
    status = getattr(state, "model_status", None)
    value = status.value if status is not None else "loading"
    if value == "loading":
        log.info("generate rejected — model still loading")
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "30"},
            content={"detail": "model is still loading, retry shortly",
                     "model_status": "loading"},
        )
    if value == "failed":
        log.warning("generate rejected — model failed: %s", state.model_error)
        return JSONResponse(
            status_code=503,
            content={"detail": "model failed to load at startup",
                     "model_status": "failed", "error": state.model_error},
        )
    return None


def _sampling_params(req: GenerateRequest):
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        seed=req.seed,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
    )


@router.post("/generate")
async def generate(request: Request, body: GenerateRequest):
    state = request.app.state
    blocked = _guard(state)
    if blocked is not None:
        return blocked
    handle = state.engine
    if handle is None:
        raise HTTPException(status_code=503, detail="engine not available")

    sp = _sampling_params(body)
    request_id = uuid4().hex
    log.info("generate start id=%s max_tokens=%d stream=%s",
             request_id, body.max_tokens, body.stream)

    if body.stream:
        return StreamingResponse(
            _sse(handle.engine, body.prompt, sp, request_id, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    final = None
    try:
        async for out in handle.engine.generate(body.prompt, sp, request_id):
            final = out
    except asyncio.CancelledError:
        await handle.engine.abort(request_id)
        raise
    if final is None:
        raise HTTPException(status_code=500, detail="no output produced")
    o = final.outputs[0]
    log.info("generate done id=%s tokens=%d reason=%s",
             request_id, len(o.token_ids), o.finish_reason)
    return {
        "request_id": request_id,
        "text": o.text,
        "finish_reason": o.finish_reason,
        "prompt_tokens": len(final.prompt_token_ids or []),
        "completion_tokens": len(o.token_ids),
    }


async def _sse(engine, prompt, sp, request_id, request: Request):
    prev = ""
    finished = False
    try:
        async for out in engine.generate(prompt, sp, request_id):
            o = out.outputs[0]
            delta = o.text[len(prev):]
            prev = o.text
            if delta:
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            if await request.is_disconnected():
                log.info("client disconnected, aborting id=%s", request_id)
                await engine.abort(request_id)
                return
            if o.finish_reason is not None:
                finished = True
                yield f"data: {json.dumps({'delta': '', 'finish_reason': o.finish_reason, 'done': True})}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        await engine.abort(request_id)
        raise
    finally:
        if not finished:
            try:
                await engine.abort(request_id)
            except Exception:  # noqa: BLE001 - best effort
                pass
